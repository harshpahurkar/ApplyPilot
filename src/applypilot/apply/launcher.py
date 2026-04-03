"""Apply orchestration: acquire jobs, spawn Claude Code sessions, track results.

This is the main entry point for the apply pipeline. It pulls jobs from
the database, launches Chrome + Claude Code for each one, parses the
result, and updates the database. Supports parallel workers via --workers.
"""

import atexit
import json
import logging
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.live import Live

from applypilot import config
from applypilot.database import get_connection
from applypilot.errors import ErrorCategory, classify_apply_error
from applypilot.apply import chrome, dashboard, prompt as prompt_mod
from applypilot.apply.stealth import (
    get_pacer, get_budget, get_rotator,
    prepare_copilot_env,
)
from applypilot.apply.chrome import (
    launch_chrome, cleanup_worker, kill_all_chrome,
    reset_worker_dir, cleanup_on_exit, _kill_process_tree,
    BASE_CDP_PORT,
)
from applypilot.apply.dashboard import (
    init_worker, update_state, add_event, get_state,
    render_full, get_totals,
)

logger = logging.getLogger(__name__)

# Blocked sites loaded from config/sites.yaml
def _load_blocked():
    from applypilot.config import load_blocked_sites
    return load_blocked_sites()

# How often to poll the DB when the queue is empty (seconds)
POLL_INTERVAL = config.DEFAULTS["poll_interval"]

# Thread-safe shutdown coordination
_stop_event = threading.Event()

# Track active Claude Code processes for skip (Ctrl+C) handling
_claude_procs: dict[int, subprocess.Popen] = {}
_claude_lock = threading.Lock()

# Register cleanup on exit
atexit.register(cleanup_on_exit)
if platform.system() != "Windows":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


# ---------------------------------------------------------------------------
# MCP config
# ---------------------------------------------------------------------------

def _make_mcp_config(cdp_port: int) -> dict:
    """Build MCP config dict for a specific CDP port."""
    return {
        "mcpServers": {
            "playwright": {
                "command": "npx",
                "args": [
                    "-y",
                    "@playwright/mcp@latest",
                    f"--cdp-endpoint=http://localhost:{cdp_port}",
                    f"--viewport-size={config.DEFAULTS['viewport']}",
                ],
            },
            "gmail": {
                "command": "npx",
                "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
            },
        }
    }


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

def acquire_job(target_url: str | None = None, min_score: int = 7,
                worker_id: int = 0) -> dict | None:
    """Atomically acquire the next job to apply to.

    Args:
        target_url: Apply to a specific URL instead of picking from queue.
        min_score: Minimum fit_score threshold.
        worker_id: Worker claiming this job (for tracking).

    Returns:
        Job dict or None if the queue is empty.
    """
    from applypilot.config import is_manual_ats

    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        manual_updates = 0

        if target_url:
            like = f"%{target_url.split('?')[0].rstrip('/')}%"
            row = conn.execute("""
                SELECT url, title, site, application_url, tailored_resume_path,
                       fit_score, location, full_description, cover_letter_path
                FROM jobs
                WHERE (url = ? OR application_url = ? OR application_url LIKE ? OR url LIKE ?)
                  AND tailored_resume_path IS NOT NULL
                  AND apply_status != 'in_progress'
                LIMIT 1
            """, (target_url, target_url, like, like)).fetchone()
        else:
            blocked_sites, blocked_patterns = _load_blocked()
            # Build parameterized filters to avoid SQL injection
            params: list = [min_score]
            site_clause = ""
            if blocked_sites:
                placeholders = ",".join("?" * len(blocked_sites))
                site_clause = f"AND site NOT IN ({placeholders})"
                params.extend(blocked_sites)
            url_clauses = ""
            if blocked_patterns:
                url_clauses = " ".join(f"AND url NOT LIKE ?" for _ in blocked_patterns)
                params.extend(blocked_patterns)
            rows = conn.execute(f"""
                SELECT url, title, site, application_url, tailored_resume_path,
                       fit_score, location, full_description, cover_letter_path,
                       CASE
                           WHEN COALESCE(application_url, url) LIKE '%greenhouse.io%' THEN 1
                           WHEN COALESCE(application_url, url) LIKE '%boards.greenhouse.io%' THEN 1
                           WHEN COALESCE(application_url, url) LIKE '%grnh.se%' THEN 1
                           WHEN COALESCE(application_url, url) LIKE '%careerpuck.com%' THEN 1
                           WHEN COALESCE(application_url, url) LIKE '%ashbyhq.com%' THEN 2
                           WHEN COALESCE(application_url, url) LIKE '%jobs.ashby.com%' THEN 2
                           WHEN COALESCE(application_url, url) LIKE '%lever.co%' THEN 3
                           WHEN COALESCE(application_url, url) LIKE '%jobs.lever.co%' THEN 3
                           WHEN COALESCE(application_url, url) LIKE '%smartrecruiters.com%' THEN 4
                           WHEN COALESCE(application_url, url) LIKE '%betterteam.com%' THEN 4
                           WHEN COALESCE(application_url, url) LIKE '%rippling.com%' THEN 4
                           WHEN COALESCE(application_url, url) LIKE '%workable.com%' THEN 4
                           WHEN COALESCE(application_url, url) LIKE '%elastic.co%' THEN 4
                           WHEN COALESCE(application_url, url) LIKE '%applytojob.com%' THEN 4
                           WHEN COALESCE(application_url, url) LIKE '%recruitingbypaycor.com%' THEN 4
                           WHEN COALESCE(application_url, url) LIKE '%myworkdayjobs.com%' THEN 7
                           WHEN COALESCE(application_url, url) LIKE '%wd5.myworkdaysite%' THEN 7
                           WHEN COALESCE(application_url, url) LIKE '%workday.com%' THEN 7
                           WHEN COALESCE(application_url, url) LIKE '%successfactors.com%' THEN 7
                           WHEN COALESCE(application_url, url) LIKE '%careers.wbd.com%' THEN 7
                           WHEN COALESCE(application_url, url) LIKE '%scotiabank.com%' THEN 7
                           WHEN COALESCE(application_url, url) LIKE '%ca.indeed.com%' THEN 6
                           ELSE 5
                       END AS portal_priority
                FROM jobs
                WHERE tailored_resume_path IS NOT NULL
                  AND (apply_status IS NULL OR apply_status = 'failed')
                  AND (apply_attempts IS NULL OR apply_attempts < ?)
                  AND fit_score >= ?
                  {site_clause}
                  {url_clauses}
                ORDER BY portal_priority ASC, fit_score DESC, url
                LIMIT 50
            """, [config.DEFAULTS["max_apply_attempts"]] + params).fetchall()

            row = None
            for candidate in rows:
                apply_url = candidate["application_url"] or candidate["url"]
                if is_manual_ats(apply_url):
                    conn.execute(
                        "UPDATE jobs SET apply_status = 'manual', apply_error = 'manual ATS' WHERE url = ?",
                        (candidate["url"],),
                    )
                    manual_updates += 1
                    continue

                resume_path = candidate["tailored_resume_path"] or ""
                resume_pdf = Path(resume_path).with_suffix(".pdf") if resume_path else None
                if not resume_pdf or not resume_pdf.exists():
                    continue

                row = candidate
                break

        if not row:
            if target_url is None and manual_updates:
                conn.commit()
                return None
            conn.rollback()
            return None

        # Skip manual ATS sites (unsolvable CAPTCHAs)
        apply_url = row["application_url"] or row["url"]
        if is_manual_ats(apply_url):
            conn.execute(
                "UPDATE jobs SET apply_status = 'manual', apply_error = 'manual ATS' WHERE url = ?",
                (row["url"],),
            )
            conn.commit()
            logger.info("Skipping manual ATS: %s", row["url"][:80])
            return None

        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            UPDATE jobs SET apply_status = 'in_progress',
                           agent_id = ?,
                           last_attempted_at = ?
            WHERE url = ?
        """, (f"worker-{worker_id}", now, row["url"]))
        conn.commit()

        return dict(row)
    except Exception:
        conn.rollback()
        raise


def mark_result(url: str, status: str, error: str | None = None,
                permanent: bool = False, duration_ms: int | None = None,
                task_id: str | None = None) -> None:
    """Update a job's apply status in the database.

    Uses structured error classification: permanent errors get attempts=99,
    transient errors increment normally. If `permanent` is not explicitly
    set, the error category is inferred from the status/error text.
    """
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    if status == "applied":
        conn.execute("""
            UPDATE jobs SET apply_status = 'applied', applied_at = ?,
                           apply_error = NULL, agent_id = NULL,
                           apply_duration_ms = ?, apply_task_id = ?
            WHERE url = ?
        """, (now, duration_ms, task_id, url))
    else:
        # Auto-classify if permanent flag not explicitly set
        if not permanent:
            category = classify_apply_error(status, error or "")
            permanent = category.is_permanent
        attempts = 99 if permanent else "COALESCE(apply_attempts, 0) + 1"
        conn.execute(f"""
            UPDATE jobs SET apply_status = ?, apply_error = ?,
                           apply_attempts = {attempts}, agent_id = NULL,
                           apply_duration_ms = ?, apply_task_id = ?
            WHERE url = ?
        """, (status, error or "unknown", duration_ms, task_id, url))
    conn.commit()


def release_lock(url: str) -> None:
    """Release the in_progress lock without changing status."""
    conn = get_connection()
    conn.execute(
        "UPDATE jobs SET apply_status = NULL, agent_id = NULL WHERE url = ? AND apply_status = 'in_progress'",
        (url,),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Utility modes (--gen, --mark-applied, --mark-failed, --reset-failed)
# ---------------------------------------------------------------------------

def gen_prompt(target_url: str, min_score: int = 7,
               model: str = "haiku", worker_id: int = 0) -> Path | None:
    """Generate a prompt file and print the Claude CLI command for manual debugging.

    Returns:
        Path to the generated prompt file, or None if no job found.
    """
    job = acquire_job(target_url=target_url, min_score=min_score, worker_id=worker_id)
    if not job:
        return None

    # Read resume text
    resume_path = job.get("tailored_resume_path")
    txt_path = Path(resume_path).with_suffix(".txt") if resume_path else None
    resume_text = ""
    if txt_path and txt_path.exists():
        resume_text = txt_path.read_text(encoding="utf-8")

    prompt = prompt_mod.build_prompt(job=job, tailored_resume=resume_text)

    # Release the lock so the job stays available
    release_lock(job["url"])

    # Write prompt file
    config.ensure_dirs()
    site_slug = (job.get("site") or "unknown")[:20].replace(" ", "_")
    prompt_file = config.LOG_DIR / f"prompt_{site_slug}_{job['title'][:30].replace(' ', '_')}.txt"
    prompt_file.write_text(prompt, encoding="utf-8")

    # Write MCP config for reference
    port = BASE_CDP_PORT + worker_id
    mcp_path = config.APP_DIR / f".mcp-apply-{worker_id}.json"
    mcp_path.write_text(json.dumps(_make_mcp_config(port)), encoding="utf-8")

    return prompt_file


def mark_job(url: str, status: str, reason: str | None = None) -> None:
    """Manually mark a job's apply status in the database.

    Args:
        url: Job URL to mark.
        status: Either 'applied' or 'failed'.
        reason: Failure reason (only for status='failed').
    """
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    if status == "applied":
        conn.execute("""
            UPDATE jobs SET apply_status = 'applied', applied_at = ?,
                           apply_error = NULL, agent_id = NULL
            WHERE url = ?
        """, (now, url))
    else:
        conn.execute("""
            UPDATE jobs SET apply_status = 'failed', apply_error = ?,
                           apply_attempts = 99, agent_id = NULL
            WHERE url = ?
        """, (reason or "manual", url))
    conn.commit()


def reset_failed() -> int:
    """Reset all failed jobs so they can be retried.

    Returns:
        Number of jobs reset.
    """
    conn = get_connection()
    cursor = conn.execute("""
        UPDATE jobs SET apply_status = NULL, apply_error = NULL,
                       apply_attempts = 0, agent_id = NULL
        WHERE apply_status = 'failed'
          OR (apply_status IS NOT NULL AND apply_status != 'applied'
              AND apply_status != 'in_progress')
    """)
    conn.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Per-job execution
# ---------------------------------------------------------------------------

def _backend_available(name: str) -> bool:
    """Return whether an apply backend is installed on this machine."""
    if name == "copilot":
        return bool(shutil.which("copilot") or shutil.which("gh"))
    return bool(shutil.which("claude"))


def _select_backend(preferred: str) -> str | None:
    """Resolve preferred backend into a concrete backend name."""
    pref = (preferred or "copilot").strip().lower()
    if pref == "auto":
        if _backend_available("copilot"):
            return "copilot"
        if _backend_available("claude"):
            return "claude"
        return None
    if pref in {"claude", "copilot"} and _backend_available(pref):
        return pref
    return None


def _copilot_apply_default_model() -> str:
    """Return the default model string to use for Copilot apply sessions."""
    override = os.environ.get("APPLYPILOT_COPILOT_APPLY_MODEL", "").strip()
    if override:
        return override

    shared = os.environ.get("COPILOT_LLM_MODEL", "").strip()
    if shared:
        return shared

    return "gpt-4.1"


def _normalize_model_for_backend(model: str, backend: str) -> str:
    """Map user-facing model aliases to backend-valid model identifiers."""
    requested = (model or "").strip()
    if backend == "claude":
        if not requested or requested.lower() == "haiku":
            return os.environ.get(
                "APPLYPILOT_CLAUDE_APPLY_MODEL", "claude-sonnet-4.6"
            )
        return requested
    if backend != "copilot":
        return requested or "haiku"

    default_model = _copilot_apply_default_model()
    if not requested:
        return default_model

    lowered = requested.lower()
    haiku_aliases = {
        "haiku",
        "claude-haiku",
        "claude_haiku",
        "claude-3-haiku",
        "claude-3.5-haiku",
        "claude-3-5-haiku",
        "claude-3-5-haiku-latest",
    }
    if lowered in haiku_aliases:
        return (
            os.environ.get("APPLYPILOT_COPILOT_MODEL_HAIKU_EQUIVALENT", "").strip()
            or default_model
        )

    return requested


def _build_agent_command(
    backend: str,
    model: str,
    mcp_config_path: Path,
    disallowed_tools: str,
    prompt_text: str | None = None,
) -> list[str] | None:
    """Build subprocess command for the selected agent backend."""
    if backend == "claude":
        claude_exe = shutil.which("claude") or "claude"
        return [
            claude_exe,
            "--model", model,
            "-p",
            "--mcp-config", str(mcp_config_path),
            "--permission-mode", "bypassPermissions",
            "--disallowedTools", disallowed_tools,
            "--no-session-persistence",
            "--output-format", "stream-json",
            "--verbose", "-",
        ]

    if backend == "copilot":
        copilot_args = [
            "--model", model,
            "--additional-mcp-config", f"@{mcp_config_path}",
            "--allow-all-tools",
            "--allow-all-paths",
            "--allow-all-urls",
            "--no-ask-user",
            "--output-format", "json",
            "-p", prompt_text or "",
        ]

        # Override when local Copilot CLI syntax differs:
        # APPLYPILOT_COPILOT_CMD="copilot --model {model} --additional-mcp-config @{mcp_config} -p {prompt}"
        custom = os.environ.get("APPLYPILOT_COPILOT_CMD", "").strip()
        if custom:
            rendered = custom.format(
                model=model,
                mcp_config=str(mcp_config_path),
                prompt=prompt_text or "",
            )
            return shlex.split(rendered, posix=(platform.system() != "Windows"))

        # Prefer invoking the npm loader directly via node on Windows.
        # This avoids wrapper (.bat/.ps1) command-line length limits.
        appdata = os.environ.get("APPDATA", "")
        node_exe = shutil.which("node")
        npm_loader = Path(appdata) / "npm" / "node_modules" / "@github" / "copilot" / "npm-loader.js"
        if platform.system() == "Windows" and node_exe and npm_loader.exists():
            return [node_exe, str(npm_loader), *copilot_args]

        copilot_exe = shutil.which("copilot")
        if copilot_exe:
            return [copilot_exe, *copilot_args]

        gh_exe = shutil.which("gh")
        if gh_exe:
            return [
                gh_exe,
                "copilot",
                "--",
                *copilot_args,
            ]

    return None

def run_job(job: dict, port: int, worker_id: int = 0,
            model: str = "haiku", dry_run: bool = False,
            agent_backend: str = "copilot") -> tuple[str, int]:
    """Spawn an agent session for one job application.

    Returns:
        Tuple of (status_string, duration_ms). Status is one of:
        'applied', 'expired', 'captcha', 'login_issue',
        'failed:reason', or 'skipped'.
    """
    # Read tailored resume text
    resume_path = job.get("tailored_resume_path")
    txt_path = Path(resume_path).with_suffix(".txt") if resume_path else None
    resume_text = ""
    if txt_path and txt_path.exists():
        resume_text = txt_path.read_text(encoding="utf-8")

    # Build the prompt
    agent_prompt = prompt_mod.build_prompt(
        job=job,
        tailored_resume=resume_text,
        dry_run=dry_run,
    )

    # Write per-worker MCP config
    mcp_config_path = config.APP_DIR / f".mcp-apply-{worker_id}.json"
    mcp_config_path.write_text(json.dumps(_make_mcp_config(port)), encoding="utf-8")

    selected_backend = _select_backend(agent_backend)
    if not selected_backend:
        duration_ms = 0
        reason = f"backend_unavailable:{agent_backend}"
        add_event(f"[W{worker_id}] {reason}")
        update_state(worker_id, status="failed", last_action=reason[:35])
        return f"failed:{reason}", duration_ms

    effective_model = _normalize_model_for_backend(model, selected_backend)
    if effective_model != model:
        add_event(
            f"[W{worker_id}] Model mapped for {selected_backend}: {model} -> {effective_model}"
        )
        logger.info(
            "Worker %d mapped model '%s' to '%s' for backend '%s'",
            worker_id,
            model,
            effective_model,
            selected_backend,
        )

    # Copilot CLI expects prompt text via -p (arg), so very large prompts can
    # hit Windows command length limits. Keep the beginning and RESULT tail.
    if selected_backend == "copilot":
        try:
            max_chars = int(os.environ.get("APPLYPILOT_COPILOT_PROMPT_MAX", "12000"))
        except ValueError:
            max_chars = 12000
        if len(agent_prompt) > max_chars:
            head = int(max_chars * 0.7)
            tail = max_chars - head
            agent_prompt = (
                agent_prompt[:head]
                + "\n\n[Prompt truncated for Copilot CLI argument limits]\n\n"
                + agent_prompt[-tail:]
            )
            add_event(f"[W{worker_id}] Copilot prompt truncated ({max_chars} chars)")
            logger.warning(
                "Worker %d: truncated Copilot prompt from %d to ~%d chars",
                worker_id,
                len(agent_prompt),
                max_chars,
            )

    disallowed_tools = (
        "mcp__gmail__draft_email,mcp__gmail__modify_email,"
        "mcp__gmail__delete_email,mcp__gmail__download_attachment,"
        "mcp__gmail__batch_modify_emails,mcp__gmail__batch_delete_emails,"
        "mcp__gmail__create_label,mcp__gmail__update_label,"
        "mcp__gmail__delete_label,mcp__gmail__get_or_create_label,"
        "mcp__gmail__list_email_labels,mcp__gmail__create_filter,"
        "mcp__gmail__list_filters,mcp__gmail__get_filter,"
        "mcp__gmail__delete_filter"
    )

    cmd = _build_agent_command(
        backend=selected_backend,
        model=effective_model,
        mcp_config_path=mcp_config_path,
        disallowed_tools=disallowed_tools,
        prompt_text=agent_prompt,
    )
    if not cmd:
        duration_ms = 0
        reason = f"backend_command_error:{selected_backend}"
        add_event(f"[W{worker_id}] {reason}")
        update_state(worker_id, status="failed", last_action=reason[:35])
        return f"failed:{reason}", duration_ms

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    if selected_backend == "copilot":
        env = prepare_copilot_env(env)
    elif selected_backend == "claude":
        # Route Claude Code through copilot-api proxy if configured
        proxy_url = os.environ.get(
            "APPLYPILOT_COPILOT_PROXY_URL", "http://localhost:4141"
        )
        env["ANTHROPIC_BASE_URL"] = proxy_url
        env["ANTHROPIC_AUTH_TOKEN"] = "dummy"
        env["DISABLE_NON_ESSENTIAL_MODEL_CALLS"] = "1"
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

    worker_dir = reset_worker_dir(worker_id)

    # Make sure worker-local copies of upload artifacts exist in the CWD.
    current_dir = config.APPLY_WORKER_DIR / "current"
    for src_file in current_dir.glob("*") if current_dir.exists() else []:
        dst_file = worker_dir / src_file.name
        if not dst_file.exists():
            shutil.copy(str(src_file), str(dst_file))

    # Rewrite prompt paths from current/ to worker-specific directory.
    old_prefix = str(current_dir).replace("\\", "/")
    new_prefix = str(worker_dir).replace("\\", "/")
    agent_prompt = agent_prompt.replace(old_prefix, new_prefix)
    agent_prompt = agent_prompt.replace(str(current_dir), str(worker_dir))

    parse_stream_json = selected_backend == "claude"

    update_state(worker_id, status="applying", job_title=job["title"],
                 company=job.get("site", ""), score=job.get("fit_score", 0),
                 start_time=time.time(), actions=0,
                 last_action=f"starting ({selected_backend})")
    add_event(
        f"[W{worker_id}] Starting ({selected_backend}): "
        f"{job['title'][:40]} @ {job.get('site', '')}"
    )

    worker_log = config.LOG_DIR / f"worker-{worker_id}.log"
    ts_header = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_header = (
        f"\n{'=' * 60}\n"
        f"[{ts_header}] {job['title']} @ {job.get('site', '')}\n"
        f"URL: {job.get('application_url') or job['url']}\n"
        f"Score: {job.get('fit_score', 'N/A')}/10\n"
        f"{'=' * 60}\n"
    )

    # Per-job wall-clock timeout (seconds). Skip and move on if exceeded.
    job_timeout = int(os.environ.get("APPLYPILOT_JOB_TIMEOUT", "420"))  # 7 min default

    start = time.time()
    stats: dict = {}
    proc = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=(subprocess.PIPE if selected_backend == "claude" else subprocess.DEVNULL),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(worker_dir),
        )
        with _claude_lock:
            _claude_procs[worker_id] = proc

        if selected_backend == "claude" and proc.stdin:
            proc.stdin.write(agent_prompt)
            proc.stdin.close()

        text_parts: list[str] = []

        def _collect_text(value: object) -> list[str]:
            out: list[str] = []
            if isinstance(value, str):
                text = value.strip()
                if text:
                    out.append(text)
                return out
            if isinstance(value, list):
                for item in value:
                    out.extend(_collect_text(item))
                return out
            if isinstance(value, dict):
                for key in ("message", "content", "text", "result", "response"):
                    if key in value:
                        out.extend(_collect_text(value[key]))
            return out

        with open(worker_log, "a", encoding="utf-8") as lf:
            lf.write(log_header)

            for line in proc.stdout:
                # Wall-clock timeout: skip job if taking too long
                if time.time() - start > job_timeout:
                    elapsed = int(time.time() - start)
                    add_event(f"[W{worker_id}] TIMEOUT ({elapsed}s): {job['title'][:30]}")
                    update_state(worker_id, status="failed",
                                 last_action=f"TIMEOUT ({elapsed}s)")
                    _kill_process_tree(proc.pid)
                    proc = None
                    return "failed:timeout", int((time.time() - start) * 1000)

                line = line.strip()
                if not line:
                    continue
                try:
                    if parse_stream_json:
                        msg = json.loads(line)
                        msg_type = msg.get("type")
                        if msg_type == "assistant":
                            for block in msg.get("message", {}).get("content", []):
                                bt = block.get("type")
                                if bt == "text":
                                    text_parts.append(block["text"])
                                    lf.write(block["text"] + "\n")
                                elif bt == "tool_use":
                                    name = (
                                        block.get("name", "")
                                        .replace("mcp__playwright__", "")
                                        .replace("mcp__gmail__", "gmail:")
                                    )
                                    inp = block.get("input", {})
                                    if "url" in inp:
                                        desc = f"{name} {inp['url'][:60]}"
                                    elif "ref" in inp:
                                        desc = f"{name} {inp.get('element', inp.get('text', ''))}"[:50]
                                    elif "fields" in inp:
                                        desc = f"{name} ({len(inp['fields'])} fields)"
                                    elif "paths" in inp:
                                        desc = f"{name} upload"
                                    else:
                                        desc = name

                                    lf.write(f"  >> {desc}\n")
                                    ws = get_state(worker_id)
                                    cur_actions = ws.actions if ws else 0
                                    update_state(worker_id,
                                                 actions=cur_actions + 1,
                                                 last_action=desc[:35])
                        elif msg_type == "result":
                            stats = {
                                "input_tokens": msg.get("usage", {}).get("input_tokens", 0),
                                "output_tokens": msg.get("usage", {}).get("output_tokens", 0),
                                "cache_read": msg.get("usage", {}).get("cache_read_input_tokens", 0),
                                "cache_create": msg.get("usage", {}).get("cache_creation_input_tokens", 0),
                                "cost_usd": msg.get("total_cost_usd", 0),
                                "turns": msg.get("num_turns", 0),
                            }
                            text_parts.append(msg.get("result", ""))
                    else:
                        # Copilot JSON stream/event output (plus plain fallback)
                        text_parts.append(line)
                        lf.write(line + "\n")
                        msg = json.loads(line)
                        if isinstance(msg, dict):
                            extracted: list[str] = []
                            if msg.get("type") == "assistant.message":
                                extracted.extend(_collect_text(msg.get("data", {})))
                            for key in ("message", "content", "text", "result", "response"):
                                if key in msg:
                                    extracted.extend(_collect_text(msg[key]))
                            for chunk in extracted:
                                if chunk and chunk != line:
                                    text_parts.append(chunk)
                                    lf.write(chunk + "\n")
                except json.JSONDecodeError:
                    text_parts.append(line)
                    lf.write(line + "\n")

        proc.wait(timeout=300)
        returncode = proc.returncode
        proc = None

        if returncode and returncode < 0:
            return "skipped", int((time.time() - start) * 1000)

        output = "\n".join(text_parts)
        elapsed = int(time.time() - start)
        duration_ms = int((time.time() - start) * 1000)

        if selected_backend == "copilot":
            out_lower = output.lower()
            if "from --model flag is not available" in out_lower:
                add_event(f"[W{worker_id}] MODEL UNAVAILABLE ({elapsed}s)")
                update_state(worker_id, status="failed", last_action="model unavailable")
                return "failed:model_unavailable", duration_ms
            if "(rate_limit)" in out_lower or ("rate limit" in out_lower and "try again" in out_lower):
                add_event(f"[W{worker_id}] RATE LIMITED ({elapsed}s)")
                update_state(worker_id, status="failed", last_action="rate limited")
                return "failed:rate_limited", duration_ms

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_log = config.LOG_DIR / f"agent_{selected_backend}_{ts}_w{worker_id}_{job.get('site', 'unknown')[:20]}.txt"
        job_log.write_text(output, encoding="utf-8")

        if stats:
            cost = stats.get("cost_usd", 0)
            ws = get_state(worker_id)
            prev_cost = ws.total_cost if ws else 0.0
            update_state(worker_id, total_cost=prev_cost + cost)

        def _clean_reason(s: str) -> str:
            return re.sub(r'[*`".\s]+$', '', s).strip()

        explicit_statuses = {
            "APPLIED": "applied",
            "SUCCESS": "applied",
            "SUBMITTED": "applied",
            "EXPIRED": "expired",
            "CAPTCHA": "captcha",
            "LOGIN_ISSUE": "login_issue",
        }
        for result_status, normalized in explicit_statuses.items():
            if f"RESULT:{result_status}" in output:
                add_event(f"[W{worker_id}] {result_status} ({elapsed}s): {job['title'][:30]}")
                update_state(worker_id, status=normalized,
                             last_action=f"{result_status} ({elapsed}s)")
                return normalized, duration_ms

        # Fuzzy fallback for models/CLIs that do not emit strict RESULT lines.
        output_lower = output.lower()
        fuzzy_map = {
            "expired": [
                r"result\s*:\s*expired",
                r"job\s+(?:is\s+)?(?:not\s+found|expired|no\s+longer\s+(?:available|accepting))",
                r"position\s+(?:has\s+been\s+)?(?:filled|closed|removed)",
                r"this\s+(?:job|listing|posting)\s+(?:is\s+)?no\s+longer",
                r"no\s+longer\s+accepting\s+appl",
                r"not\s+accepting\s+appl",
                r"currently\s+closed",
                r"(?:is|was)\s+(?:no longer|not)\s+(?:available|open|accepting)",
                r"cannot\s+proceed.*appl",
            ],
            "captcha": [r"result\s*:\s*captcha", r"captcha\s+detected"],
            "login_issue": [r"result\s*:\s*login.?issue", r"login\s+(?:required|wall)"],
            "applied": [
                r"result\s*:\s*applied",
                r"result\s*:\s*(?:success|submitted)",
            ],
        }
        for fuzzy_status, patterns in fuzzy_map.items():
            for pat in patterns:
                if re.search(pat, output_lower):
                    result_status = fuzzy_status.upper()
                    add_event(f"[W{worker_id}] {result_status} ({elapsed}s): {job['title'][:30]}")
                    update_state(worker_id, status=fuzzy_status,
                                 last_action=f"{result_status} ({elapsed}s)")
                    return fuzzy_status, duration_ms

        if "RESULT:FAILED" in output:
            for out_line in output.split("\n"):
                if "RESULT:FAILED" in out_line:
                    reason = (
                        out_line.split("RESULT:FAILED:")[-1].strip()
                        if ":" in out_line[out_line.index("FAILED") + 6:]
                        else "unknown"
                    )
                    reason = _clean_reason(reason)
                    PROMOTE_TO_STATUS = {"captcha", "expired", "login_issue"}
                    if reason in PROMOTE_TO_STATUS:
                        add_event(f"[W{worker_id}] {reason.upper()} ({elapsed}s): {job['title'][:30]}")
                        update_state(worker_id, status=reason,
                                     last_action=f"{reason.upper()} ({elapsed}s)")
                        return reason, duration_ms
                    add_event(f"[W{worker_id}] FAILED ({elapsed}s): {reason[:30]}")
                    update_state(worker_id, status="failed",
                                 last_action=f"FAILED: {reason[:25]}")
                    return f"failed:{reason}", duration_ms
            return "failed:unknown", duration_ms

        add_event(f"[W{worker_id}] NO RESULT ({elapsed}s)")
        update_state(worker_id, status="failed", last_action=f"no result ({elapsed}s)")
        return "failed:no_result_line", duration_ms

    except subprocess.TimeoutExpired:
        duration_ms = int((time.time() - start) * 1000)
        elapsed = int(time.time() - start)
        add_event(f"[W{worker_id}] TIMEOUT ({elapsed}s)")
        update_state(worker_id, status="failed", last_action=f"TIMEOUT ({elapsed}s)")
        return "failed:timeout", duration_ms
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        add_event(f"[W{worker_id}] ERROR: {str(e)[:40]}")
        update_state(worker_id, status="failed", last_action=f"ERROR: {str(e)[:25]}")
        return f"failed:{str(e)[:100]}", duration_ms
    finally:
        with _claude_lock:
            _claude_procs.pop(worker_id, None)
        if proc is not None and proc.poll() is None:
            _kill_process_tree(proc.pid)


# ---------------------------------------------------------------------------
# Permanent failure classification
# ---------------------------------------------------------------------------

PERMANENT_FAILURES: set[str] = {
    "expired", "captcha", "login_issue", "timeout",
    "not_eligible_location", "not_eligible_salary",
    "already_applied", "account_required",
    "not_a_job_application", "unsafe_permissions",
    "unsafe_verification", "sso_required",
    "site_blocked", "cloudflare_blocked", "blocked_by_cloudflare",
}

PERMANENT_PREFIXES: tuple[str, ...] = ("site_blocked", "cloudflare", "blocked_by")


def _is_permanent_failure(result: str) -> bool:
    """Determine if a failure should never be retried."""
    reason = result.split(":", 1)[-1] if ":" in result else result
    return (
        result in PERMANENT_FAILURES
        or reason in PERMANENT_FAILURES
        or any(reason.startswith(p) for p in PERMANENT_PREFIXES)
    )


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def worker_loop(worker_id: int = 0, limit: int = 1,
                target_url: str | None = None,
                min_score: int = 7, headless: bool = False,
                model: str = "haiku", dry_run: bool = False,
                agent_backend: str = "copilot",
                persistent: bool = False) -> tuple[int, int]:
    """Run jobs sequentially until limit is reached or queue is empty.

    Args:
        worker_id: Numeric worker identifier.
        limit: Max jobs to process (0 = continuous).
        target_url: Apply to a specific URL.
        min_score: Minimum fit_score threshold.
        headless: Run Chrome headless.
        model: Agent model name.
        dry_run: Don't click Submit.
        agent_backend: Apply agent backend (claude, copilot, auto).
        persistent: If True, retry transient failures with exponential
                    backoff instead of moving to the next job.

    Returns:
        Tuple of (applied_count, failed_count).
    """
    applied = 0
    failed = 0
    continuous = limit == 0
    jobs_done = 0
    empty_polls = 0
    port = BASE_CDP_PORT + worker_id
    rate_limit_backoff = 0  # exponential backoff seconds for rate limits
    pacer = get_pacer()
    budget = get_budget()
    rotator = get_rotator()

    while not _stop_event.is_set():
        if not continuous and jobs_done >= limit:
            break

        update_state(worker_id, status="idle", job_title="", company="",
                     last_action="waiting for job", actions=0)

        job = acquire_job(target_url=target_url, min_score=min_score,
                          worker_id=worker_id)
        if not job:
            if not continuous:
                add_event(f"[W{worker_id}] Queue empty")
                update_state(worker_id, status="done", last_action="queue empty")
                break
            empty_polls += 1
            update_state(worker_id, status="idle",
                         last_action=f"polling ({empty_polls})")
            if empty_polls == 1:
                add_event(f"[W{worker_id}] Queue empty, polling every {POLL_INTERVAL}s...")
            # Use Event.wait for interruptible sleep
            if _stop_event.wait(timeout=POLL_INTERVAL):
                break  # Stop was requested during wait
            continue

        empty_polls = 0

        # ── Stealth: human-like inter-job pacing + budget check ──
        if jobs_done > 0 and agent_backend in ("copilot", "auto"):
            # Wait if request budget is exhausted for this time window
            waited = budget.wait_if_exhausted()
            if waited > 0:
                add_event(f"[W{worker_id}] Budget cooldown {waited:.0f}s")

            # Add adaptive slowdown when approaching budget limits
            adaptive = budget.adaptive_delay()
            if adaptive > 0:
                update_state(worker_id, status="pacing",
                             last_action=f"adaptive delay {adaptive:.0f}s")

            # Human-like pause between jobs
            pacer.sleep_inter_job()

        # Stealth: rotate to a free-tier model if user didn't pin one
        effective_model = model
        if agent_backend in ("copilot", "auto") and rotator.enabled:
            if model.lower() in ("haiku", "auto", ""):
                effective_model = rotator.next_model()
                if effective_model != model:
                    add_event(f"[W{worker_id}] Model rotated: {effective_model}")

        chrome_proc = None
        try:
            add_event(f"[W{worker_id}] Launching Chrome...")
            chrome_proc = launch_chrome(worker_id, port=port, headless=headless)

            result, duration_ms = run_job(
                job,
                port=port,
                worker_id=worker_id,
                model=effective_model,
                dry_run=dry_run,
                agent_backend=agent_backend,
            )

            # Record this request against the budget tracker
            budget.record()

            if result == "skipped":
                release_lock(job["url"])
                add_event(f"[W{worker_id}] Skipped: {job['title'][:30]}")
                continue
            elif result == "applied":
                mark_result(job["url"], "applied", duration_ms=duration_ms)
                applied += 1
                rate_limit_backoff = 0  # reset backoff on success
                update_state(worker_id, jobs_applied=applied,
                             jobs_done=applied + failed)
            else:
                reason = result.split(":", 1)[-1] if ":" in result else result
                is_perm = _is_permanent_failure(result)
                mark_result(job["url"], "failed", reason,
                            permanent=is_perm,
                            duration_ms=duration_ms)
                failed += 1
                update_state(worker_id, jobs_failed=failed,
                             jobs_done=applied + failed)

                # Persistent mode: on transient failure, sleep and retry
                # instead of moving to the next job
                if persistent and not is_perm and "rate_limited" not in result:
                    transient_backoff = min(30 * (2 ** min(failed - 1, 4)), 300)
                    add_event(
                        f"[W{worker_id}] Transient failure, retrying in "
                        f"{transient_backoff}s (persistent mode)"
                    )
                    update_state(
                        worker_id, status="cooldown",
                        last_action=f"transient backoff {transient_backoff}s",
                    )
                    if _stop_event.wait(timeout=transient_backoff):
                        break
                    # Don't count towards jobs_done so we re-acquire
                    jobs_done -= 1
                    continue

                # Exponential backoff on rate limits
                if "rate_limited" in result:
                    rate_limit_backoff = min(
                        (rate_limit_backoff or 30) * 2, 300
                    )
                    add_event(
                        f"[W{worker_id}] Rate limited, backing off "
                        f"{rate_limit_backoff}s"
                    )
                    update_state(
                        worker_id, status="cooldown",
                        last_action=f"rate-limit backoff {rate_limit_backoff}s",
                    )
                    if _stop_event.wait(timeout=rate_limit_backoff):
                        break
                else:
                    rate_limit_backoff = 0  # reset on non-rate-limit result

        except KeyboardInterrupt:
            release_lock(job["url"])
            if _stop_event.is_set():
                break
            add_event(f"[W{worker_id}] Job skipped (Ctrl+C)")
            continue
        except Exception as e:
            logger.exception("Worker %d launcher error", worker_id)
            add_event(f"[W{worker_id}] Launcher error: {str(e)[:40]}")
            release_lock(job["url"])
            failed += 1
            update_state(worker_id, jobs_failed=failed)
        finally:
            if chrome_proc:
                cleanup_worker(worker_id, chrome_proc)

        jobs_done += 1
        if target_url:
            break

    update_state(worker_id, status="done", last_action="finished")
    return applied, failed


# ---------------------------------------------------------------------------
# Main entry point (called from cli.py)
# ---------------------------------------------------------------------------

def main(limit: int = 1, target_url: str | None = None,
         min_score: int = 7, headless: bool = False, model: str = "haiku",
         dry_run: bool = False, continuous: bool = False,
         poll_interval: int = 60, workers: int = 1,
         agent_backend: str = "copilot",
         persistent: bool = False) -> None:
    """Launch the apply pipeline.

    Args:
        limit: Max jobs to apply to (0 or with continuous=True means run forever).
        target_url: Apply to a specific URL.
        min_score: Minimum fit_score threshold.
        headless: Run Chrome in headless mode.
        model: Agent model name.
        dry_run: Don't click Submit.
        continuous: Run forever, polling for new jobs.
        poll_interval: Seconds between DB polls when queue is empty.
        workers: Number of parallel workers (default 1).
        agent_backend: Apply agent backend (claude, copilot, auto).
        persistent: Retry transient failures with exponential backoff.
    """
    global POLL_INTERVAL
    POLL_INTERVAL = poll_interval
    _stop_event.clear()

    config.ensure_dirs()
    console = Console()

    if continuous:
        effective_limit = 0
        mode_label = "continuous"
    else:
        effective_limit = limit
        mode_label = f"{limit} jobs"

    # Initialize dashboard for all workers
    for i in range(workers):
        init_worker(i)

    worker_label = f"{workers} worker{'s' if workers > 1 else ''}"
    console.print(
        f"Launching apply pipeline ({mode_label}, {worker_label}, "
        f"backend={agent_backend}, poll every {POLL_INTERVAL}s)..."
    )
    console.print("[dim]Ctrl+C = skip current job(s) | Ctrl+C x2 = stop[/dim]")

    # Double Ctrl+C handler
    _ctrl_c_count = 0

    def _sigint_handler(sig, frame):
        nonlocal _ctrl_c_count
        _ctrl_c_count += 1
        if _ctrl_c_count == 1:
            console.print("\n[yellow]Skipping current job(s)... (Ctrl+C again to STOP)[/yellow]")
            # Kill all active Claude processes to skip current jobs
            with _claude_lock:
                for wid, cproc in list(_claude_procs.items()):
                    if cproc.poll() is None:
                        _kill_process_tree(cproc.pid)
        else:
            console.print("\n[red bold]STOPPING[/red bold]")
            _stop_event.set()
            with _claude_lock:
                for wid, cproc in list(_claude_procs.items()):
                    if cproc.poll() is None:
                        _kill_process_tree(cproc.pid)
            kill_all_chrome()
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        with Live(render_full(), console=console, refresh_per_second=2) as live:
            # Daemon thread for display refresh only (no business logic)
            _dashboard_running = True

            def _refresh():
                while _dashboard_running:
                    live.update(render_full())
                    time.sleep(0.5)

            refresh_thread = threading.Thread(target=_refresh, daemon=True)
            refresh_thread.start()

            if workers == 1:
                # Single worker — run directly in main thread
                total_applied, total_failed = worker_loop(
                    worker_id=0,
                    limit=effective_limit,
                    target_url=target_url,
                    min_score=min_score,
                    headless=headless,
                    model=model,
                    dry_run=dry_run,
                    agent_backend=agent_backend,
                    persistent=persistent,
                )
            else:
                # Multi-worker — distribute limit across workers
                if effective_limit:
                    base = effective_limit // workers
                    extra = effective_limit % workers
                    limits = [base + (1 if i < extra else 0)
                              for i in range(workers)]
                else:
                    limits = [0] * workers  # continuous mode

                with ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="apply-worker") as executor:
                    futures = {
                        executor.submit(
                            worker_loop,
                            worker_id=i,
                            limit=limits[i],
                            target_url=target_url,
                            min_score=min_score,
                            headless=headless,
                            model=model,
                            dry_run=dry_run,
                            agent_backend=agent_backend,
                            persistent=persistent,
                        ): i
                        for i in range(workers)
                    }

                    results: list[tuple[int, int]] = []
                    for future in as_completed(futures):
                        wid = futures[future]
                        try:
                            results.append(future.result())
                        except Exception:
                            logger.exception("Worker %d crashed", wid)
                            results.append((0, 0))

                total_applied = sum(r[0] for r in results)
                total_failed = sum(r[1] for r in results)

            _dashboard_running = False
            refresh_thread.join(timeout=2)
            live.update(render_full())

        totals = get_totals()
        console.print(
            f"\n[bold]Done: {total_applied} applied, {total_failed} failed "
            f"(${totals['cost']:.3f})[/bold]"
        )
        console.print(f"Logs: {config.LOG_DIR}")

    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        kill_all_chrome()
