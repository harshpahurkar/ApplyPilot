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
from applypilot.apply import chrome, dashboard, prompt as prompt_mod
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


# ---------------------------------------------------------------------------
# Role / seniority filtering
# ---------------------------------------------------------------------------

# Always skip these seniority levels (waste of time for a new-grad)
_SKIP_TITLE_KEYWORDS = {
    "senior", "sr.", "sr ", "staff", "principal", "distinguished",
    "director", "vp ", "vice president", "head of", "chief",
    "architect",  # almost always senior-level
}

# Internship / co-op / student keywords
_INTERN_KEYWORDS = {"intern", "internship", "co-op", "coop", "student", "stagiaire"}


# Patterns that indicate a job is in Canada
_CANADA_PATTERNS = (
    "canada", ", on,", ", on ", ", bc,", ", bc ", ", ab,", ", ab ",
    ", qc,", ", qc ", ", mb,", ", mb ", ", sk,", ", sk ",
    ", ns,", ", ns ", ", nb,", ", nb ", ", nl,", ", nl ",
    ", pe,", ", pe ",
    "ontario", "quebec", "british columbia", "alberta",
    "manitoba", "saskatchewan", "nova scotia",
    "toronto", "montreal", "vancouver", "ottawa", "calgary",
    "edmonton", "winnipeg", "waterloo", "kitchener", "mississauga",
    "brampton", "markham", "hamilton", "london, on", "halifax",
    "victoria, bc", "saskatoon", "regina",
)


def _should_skip_role(title: str, location: str | None) -> str | None:
    """Return a skip reason if this job should be filtered out, else None.

    Rules:
      1. Senior / staff / principal / director / lead-architect -> always skip
      2. Any job not in Canada and not remote -> skip
      3. Intern / co-op not in Canada and not remote -> skip
      4. Everything else -> keep
    """
    t = title.lower()
    loc = (location or "").lower()

    # Rule 1: Skip senior-level roles
    for kw in _SKIP_TITLE_KEYWORDS:
        if kw in t:
            return f"seniority:{kw.strip()}"

    # Rule 2: Canada-only filter — skip any non-remote job outside Canada
    is_remote = any(x in loc for x in ("remote", "anywhere", "work from home"))
    in_canada = any(x in loc for x in _CANADA_PATTERNS)
    if not is_remote and not in_canada and loc:
        return "location:not_canada_not_remote"

    # Rule 3: Skip US-only remote jobs — "Remote - US", "Remote - Seattle", etc.
    # Global remote (Ireland, EU, UK, etc.) is acceptable.
    if is_remote and loc:
        us_only_remote = (
            "remote - us", "remote (us", "us remote", "us only", "usa only",
            "remote - sf", "remote - bay area", "remote - new york", "remote - seattle",
            "remote - boston", "remote - chicago", "remote - austin", "remote - denver",
            "remote - atlanta", "remote - dallas", "remote - miami", "remote - la",
            "remote - los angeles", "remote - california", "remote - texas",
            "(seattle, wa", "(san francisco", "(new york, ny", "(chicago, il",
            "remote - washington", "remote - colorado", "remote - georgia",
        )
        if any(x in loc for x in us_only_remote) and not in_canada:
            return "location:us_remote_only"

    return None  # keep


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
    """Build Claude Code MCP config dict for a specific CDP port."""
    return {
        "mcpServers": {
            "playwright": {
                "command": "npx",
                "args": [
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

    # Loop to skip ineligible jobs (manual ATS, wrong seniority, etc.)
    # Each iteration acquires the DB lock, checks one candidate, and either
    # claims it or marks it skipped and tries again.
    max_skips = 200  # safety valve
    for _ in range(max_skips):
        conn = get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")

            if target_url:
                like = f"%{target_url.split('?')[0].rstrip('/')}%"
                row = conn.execute("""
                    SELECT url, title, site, application_url, tailored_resume_path,
                           fit_score, location, full_description, cover_letter_path
                    FROM jobs
                    WHERE (url = ? OR application_url = ? OR application_url LIKE ? OR url LIKE ?)
                      AND tailored_resume_path IS NOT NULL
                      AND (apply_status IS NULL OR apply_status != 'in_progress')
                    LIMIT 1
                """, (target_url, target_url, like, like)).fetchone()
            else:
                blocked_sites, blocked_patterns = _load_blocked()
                site_filter = " AND ".join(f"site != '{s}'" for s in blocked_sites) if blocked_sites else "1=1"
                # Filter on the URL we'll ACTUALLY navigate to (application_url if available)
                apply_url_filter = " AND ".join(
                    f"COALESCE(application_url, url) NOT LIKE '{p}'" for p in blocked_patterns
                ) if blocked_patterns else "1=1"
                row = conn.execute(f"""
                    SELECT url, title, site, application_url, tailored_resume_path,
                           fit_score, location, full_description, cover_letter_path
                    FROM jobs
                    WHERE tailored_resume_path IS NOT NULL
                      AND (apply_status IS NULL OR apply_status IN ('ready', 'failed'))
                      AND (apply_attempts IS NULL OR apply_attempts < {config.DEFAULTS["max_apply_attempts"]})
                      AND fit_score >= ?
                      AND (site != 'linkedin'
                           OR (application_url IS NOT NULL
                               AND application_url != ''
                               AND application_url NOT LIKE '%linkedin.com%'))
                      AND ({site_filter})
                      AND ({apply_url_filter})
                      AND NOT EXISTS (
                          SELECT 1 FROM jobs j2
                          WHERE j2.application_url = COALESCE(jobs.application_url, jobs.url)
                            AND j2.url != jobs.url
                            AND j2.application_url IS NOT NULL
                            AND j2.application_url != ''
                            AND (j2.apply_attempts >= {config.DEFAULTS["max_apply_attempts"]}
                                 OR j2.apply_status IN ('applied', 'skipped'))
                      )
                    ORDER BY
                      -- Tier 0: Known ATS with standard forms (best success rate) --
                      CASE
                        WHEN COALESCE(application_url, url) LIKE '%ashbyhq.com%'
                             OR COALESCE(application_url, url) LIKE '%greenhouse.io%'
                             OR COALESCE(application_url, url) LIKE '%grnh.se%'
                             OR COALESCE(application_url, url) LIKE '%smartrecruiters.com%'
                             OR COALESCE(application_url, url) LIKE '%workable.com%'
                             OR COALESCE(application_url, url) LIKE '%teamtailor.com%'
                             OR COALESCE(application_url, url) LIKE '%jobvite.com%'
                             OR COALESCE(application_url, url) LIKE '%bamboohr.com%'
                             OR COALESCE(application_url, url) LIKE '%rippling.com%'
                             OR COALESCE(application_url, url) LIKE '%pinpointhq.com%'
                             OR COALESCE(application_url, url) LIKE '%breezy.hr%'
                             OR COALESCE(application_url, url) LIKE '%applytojob%'
                             OR COALESCE(application_url, url) LIKE '%humi.ca%'
                        THEN 0
                        -- Tier 1: Other employer sites --
                        WHEN site != 'indeed' THEN 1
                        -- Tier 2: Workday/Indeed need special handling, often expired --
                        WHEN COALESCE(application_url, url) LIKE '%workday%'
                        THEN 2
                        -- Tier 3: Indeed (mostly expired, captcha-heavy) --
                        ELSE 3
                      END,
                      -- Freshness first: newest jobs get applied to first --
                      COALESCE(date_posted, discovered_at) DESC,
                      fit_score DESC
                    LIMIT 1
                """, (min_score,)).fetchone()

            if not row:
                conn.rollback()
                return None

            # --- Check 1: Manual ATS (unsolvable CAPTCHAs) ---
            apply_url = row["application_url"] or row["url"]
            if is_manual_ats(apply_url):
                conn.execute(
                    "UPDATE jobs SET apply_status = 'manual', apply_error = 'manual ATS' WHERE url = ?",
                    (row["url"],),
                )
                conn.commit()
                logger.info("Skipping manual ATS: %s", row["url"][:80])
                continue  # try next job

            # --- Check 1b: LinkedIn apply URLs / broken apply URLs ---
            # Skip ONLY if the application URL itself points to LinkedIn
            # (requires login) or is broken.  Do NOT skip jobs that were
            # *discovered* on LinkedIn but have an external ATS apply URL.
            apply_is_linkedin = "linkedin.com" in (apply_url or "").lower()
            apply_is_broken = "error=true" in (apply_url or "").lower()
            apply_has_no_url = not apply_url or apply_url.strip() == ""
            if (not target_url
                    and (apply_is_linkedin or apply_is_broken or apply_has_no_url)):
                conn.execute(
                    "UPDATE jobs SET apply_status = 'skipped', apply_error = ? WHERE url = ?",
                    ("linkedin_source", row["url"]),
                )
                conn.commit()
                logger.info("Skipping LinkedIn/broken apply URL: %s", apply_url[:80])
                continue  # try next job

            # --- Check 2: Role seniority / internship filter ---
            skip_reason = _should_skip_role(row["title"], row["location"])
            if skip_reason and not target_url:
                conn.execute(
                    "UPDATE jobs SET apply_status = 'skipped', apply_error = ? WHERE url = ?",
                    (skip_reason, row["url"]),
                )
                conn.commit()
                logger.info("Skipping [%s]: %s", skip_reason, row["title"][:60])
                continue  # try next job

            # --- All checks passed: claim this job ---
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

    logger.warning("Exhausted %d skip iterations", max_skips)
    return None


def mark_result(url: str, status: str, error: str | None = None,
                permanent: bool = False, duration_ms: int | None = None,
                task_id: str | None = None) -> None:
    """Update a job's apply status in the database."""
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


def _cleanup_dead_job(job: dict) -> None:
    """Delete an expired/404 job's resume files and DB record entirely.

    This removes:
    - Tailored resume PDF + .tex + .txt
    - Cover letter PDF + .tex + .txt
    - The job row from the database
    """
    url = job["url"]
    title = job.get("title", "")[:40]
    deleted_files = 0

    # Delete resume files
    for path_key in ("tailored_resume_path", "cover_letter_path"):
        base_path = job.get(path_key)
        if not base_path:
            continue
        base = Path(base_path)
        for ext in ("", ".tex", ".txt"):
            p = base.with_suffix(ext) if ext else base
            try:
                if p.exists():
                    p.unlink()
                    deleted_files += 1
            except Exception:
                pass
        # Also try without suffix replacement (PDF already has .pdf)
        try:
            if base.exists():
                base.unlink()
                deleted_files += 1
        except Exception:
            pass

    # Delete from database
    conn = get_connection()
    conn.execute("DELETE FROM jobs WHERE url = ?", (url,))
    conn.commit()
    logger.info("Cleaned expired job: %s (%d files deleted)", title, deleted_files)


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

def _proxy_available() -> bool:
    """Check whether the local API proxy is reachable."""
    _url = os.environ.get("APPLYPILOT_PROXY_URL", "http://localhost:4000")
    try:
        import urllib.request
        urllib.request.urlopen(f"{_url}/health", timeout=2)
        return True
    except Exception:
        return False


def _backend_available(name: str) -> bool:
    """Return whether an apply backend is installed on this machine."""
    if name == "copilot":
        return bool(shutil.which("copilot") or shutil.which("gh"))
    return bool(shutil.which("claude"))


def _select_backend(preferred: str) -> str | None:
    """Resolve preferred backend into a concrete backend name."""
    pref = (preferred or "claude").strip().lower()
    if pref == "auto":
        if _backend_available("claude"):
            return "claude"
        if _backend_available("copilot"):
            return "copilot"
        return None
    if pref in {"claude", "copilot"} and _backend_available(pref):
        return pref
    return None


def _build_agent_command(
    backend: str,
    model: str,
    mcp_config_path: Path,
    disallowed_tools: str,
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
        # Override when local Copilot CLI syntax differs:
        # APPLYPILOT_COPILOT_CMD="copilot run --model {model} --mcp-config {mcp_config} -"
        custom = os.environ.get("APPLYPILOT_COPILOT_CMD", "").strip()
        if custom:
            rendered = custom.format(model=model, mcp_config=str(mcp_config_path))
            return shlex.split(rendered, posix=(platform.system() != "Windows"))

        copilot_exe = shutil.which("copilot")
        if copilot_exe:
            return [
                copilot_exe,
                "run",
                "--model", model,
                "--mcp-config", str(mcp_config_path),
                "-",
            ]

        gh_exe = shutil.which("gh")
        if gh_exe:
            return [
                gh_exe,
                "copilot",
                "agent",
                "--model", model,
                "--mcp-config", str(mcp_config_path),
                "-",
            ]

    return None


def run_job(job: dict, port: int, worker_id: int = 0,
            model: str = "haiku", dry_run: bool = False,
        use_proxy: bool = False,
        agent_backend: str = "claude") -> tuple[str, int]:
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

    # Gmail tools that must never be used (safety: read-only Gmail access)
    _disallowed = ",".join([
        "mcp__gmail__draft_email",
        "mcp__gmail__modify_email",
        "mcp__gmail__delete_email",
        "mcp__gmail__download_attachment",
        "mcp__gmail__batch_modify_emails",
        "mcp__gmail__batch_delete_emails",
        "mcp__gmail__create_label",
        "mcp__gmail__update_label",
        "mcp__gmail__delete_label",
        "mcp__gmail__get_or_create_label",
        "mcp__gmail__list_email_labels",
        "mcp__gmail__create_filter",
        "mcp__gmail__list_filters",
        "mcp__gmail__get_filter",
        "mcp__gmail__delete_filter",
    ])

    cmd = _build_agent_command(
        backend=selected_backend,
        model=model,
        mcp_config_path=mcp_config_path,
        disallowed_tools=_disallowed,
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

    # Only route through proxy when explicitly requested (fallback mode)
    if use_proxy and selected_backend == "claude":
        _PROXY_URL = os.environ.get("APPLYPILOT_PROXY_URL", "http://localhost:4000")
        try:
            import urllib.request
            urllib.request.urlopen(f"{_PROXY_URL}/health", timeout=2)
            env["ANTHROPIC_BASE_URL"] = _PROXY_URL
            env["ANTHROPIC_API_KEY"] = "proxy-passthrough"
            logger.info("Using proxy fallback for worker %d", worker_id)
        except Exception:
            pass  # Proxy not available — fall through to direct Claude

    worker_dir = reset_worker_dir(worker_id)

    # Pre-copy resume/cover-letter into the worker directory so
    # Playwright MCP (which restricts uploads to CWD) can access them.
    _current_dir = config.APPLY_WORKER_DIR / "current"
    for src_file in _current_dir.glob("*") if _current_dir.exists() else []:
        dst_file = worker_dir / src_file.name
        if not dst_file.exists():
            shutil.copy(str(src_file), str(dst_file))
    # Rewrite absolute paths in the prompt so the agent finds local copies
    old_prefix = str(_current_dir).replace("\\", "/")
    new_prefix = str(worker_dir).replace("\\", "/")
    agent_prompt = agent_prompt.replace(old_prefix, new_prefix)
    agent_prompt = agent_prompt.replace(str(_current_dir), str(worker_dir))

    parse_stream_json = selected_backend == "claude"

    update_state(worker_id, status="applying", job_title=job["title"],
                 company=job.get("site", ""), score=job.get("fit_score", 0),
                 start_time=time.time(), actions=0, last_action=f"starting ({selected_backend})")
    add_event(f"[W{worker_id}] Starting ({selected_backend}): {job['title'][:40]} @ {job.get('site', '')}")

    worker_log = config.LOG_DIR / f"worker-{worker_id}.log"
    ts_header = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_header = (
        f"\n{'=' * 60}\n"
        f"[{ts_header}] {job['title']} @ {job.get('site', '')}\n"
        f"URL: {job.get('application_url') or job['url']}\n"
        f"Score: {job.get('fit_score', 'N/A')}/10\n"
        f"{'=' * 60}\n"
    )

    start = time.time()
    stats: dict = {}
    proc = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
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

        proc.stdin.write(agent_prompt)
        proc.stdin.close()

        # Hard deadline: kill the process if it exceeds this (seconds)
        _JOB_TIMEOUT = 420

        text_parts: list[str] = []
        timed_out = False
        with open(worker_log, "a", encoding="utf-8") as lf:
            lf.write(log_header)

            for line in proc.stdout:
                # Check hard deadline
                if time.time() - start > _JOB_TIMEOUT:
                    lf.write(f"\n  >> TIMEOUT after {_JOB_TIMEOUT}s — killing process\n")
                    timed_out = True
                    _kill_process_tree(proc.pid)
                    break

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
                        text_parts.append(line)
                        lf.write(line + "\n")
                except json.JSONDecodeError:
                    text_parts.append(line)
                    lf.write(line + "\n")

        proc.wait(timeout=30)
        returncode = proc.returncode
        proc = None

        if timed_out:
            duration_ms = int((time.time() - start) * 1000)
            elapsed = int(time.time() - start)
            add_event(f"[W{worker_id}] TIMEOUT ({elapsed}s): {job['title'][:30]}")
            update_state(worker_id, status="failed", last_action=f"TIMEOUT ({elapsed}s)")
            return "failed:timeout", duration_ms

        if returncode and returncode < 0:
            return "skipped", int((time.time() - start) * 1000)

        output = "\n".join(text_parts)
        elapsed = int(time.time() - start)
        duration_ms = int((time.time() - start) * 1000)

        # Detect Claude Code rate limit — stop pipeline immediately
        out_lower = output.lower()
        if ("hit your limit" in out_lower
                or "out of extra usage" in out_lower
                or "out of usage" in out_lower
                or "rate limit" in out_lower):
            add_event(f"[W{worker_id}] RATE LIMITED — stopping pipeline")
            update_state(worker_id, status="rate_limited",
                         last_action="rate limited — stopping")
            return "rate_limited", duration_ms

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

        for result_status in ["APPLIED", "EXPIRED", "CAPTCHA", "LOGIN_ISSUE"]:
            if f"RESULT:{result_status}" in output:
                add_event(f"[W{worker_id}] {result_status} ({elapsed}s): {job['title'][:30]}")
                update_state(worker_id, status=result_status.lower(),
                             last_action=f"{result_status} ({elapsed}s)")
                return result_status.lower(), duration_ms

        # Fuzzy fallback for non-Claude models (Gemini/GPT) that use
        # variant formatting like **Result: EXPIRED** or natural language
        _output_lower = output.lower()
        _fuzzy_map = {
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
                r"application\s+(?:has\s+been\s+)?(?:submitted|received|sent)",
                r"thank\s+you\s+for\s+(?:your\s+)?(?:applying|application|interest)",
                r"successfully\s+(?:submitted|applied)",
            ],
        }
        for fuzzy_status, patterns in _fuzzy_map.items():
            for pat in patterns:
                if re.search(pat, _output_lower):
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
    "expired",
    "not_eligible_location", "not_eligible_salary",
    "already_applied", "account_required",
    "not_a_job_application", "unsafe_permissions",
    "unsafe_verification", "sso_required",
    "site_blocked", "cloudflare_blocked", "blocked_by_cloudflare",
    "host_unreachable", "page_load_failed",
    "browser_connection_error",
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
                agent_backend: str = "claude") -> tuple[int, int]:
    """Run jobs sequentially until limit is reached or queue is empty.

    Args:
        worker_id: Numeric worker identifier.
        limit: Max jobs to process (0 = continuous).
        target_url: Apply to a specific URL.
        min_score: Minimum fit_score threshold.
        headless: Run Chrome headless.
        model: Claude model name.
        dry_run: Don't click Submit.
        agent_backend: Apply agent backend (claude, copilot, auto).

    Returns:
        Tuple of (applied_count, failed_count).
    """
    applied = 0
    failed = 0
    continuous = limit == 0
    jobs_done = 0
    empty_polls = 0
    port = BASE_CDP_PORT + worker_id

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

        chrome_proc = None
        try:
            add_event(f"[W{worker_id}] Launching Chrome...")
            chrome_proc = launch_chrome(worker_id, port=port, headless=headless)

            result, duration_ms = run_job(
                job, port=port, worker_id=worker_id,
                model=model, dry_run=dry_run, use_proxy=False,
                agent_backend=agent_backend)

            if result == "skipped":
                release_lock(job["url"])
                add_event(f"[W{worker_id}] Skipped: {job['title'][:30]}")
                continue
            elif result == "rate_limited":
                # Claude usage cap hit — release lock and stop all workers.
                # Usage resets on a per-period basis; no point retrying.
                release_lock(job["url"])
                add_event(f"[W{worker_id}] Rate limited — stopping apply workers")
                update_state(worker_id, status="rate_limited",
                             last_action="rate limited — stopped")
                _stop_event.set()
                break
            elif result == "expired":
                # Expired/closed job — mark permanently, don't retry
                mark_result(job["url"], "expired", "expired",
                            permanent=True, duration_ms=duration_ms)
                add_event(f"[W{worker_id}] EXPIRED: {job['title'][:30]}")
                update_state(worker_id, last_action="expired — skipped")
                continue  # don't count as failure
            elif result in ("captcha",):
                # Captcha — retryable (improved hCaptcha fallback chain may work on retry)
                mark_result(job["url"], "failed", result,
                            permanent=False, duration_ms=duration_ms)
                failed += 1
                update_state(worker_id, jobs_failed=failed,
                             jobs_done=applied + failed)
            elif result == "login_issue":
                # Login issue — retryable now that Gmail MCP is available
                mark_result(job["url"], "failed", result,
                            permanent=False, duration_ms=duration_ms)
                failed += 1
                update_state(worker_id, jobs_failed=failed,
                             jobs_done=applied + failed)
            elif result == "applied":
                mark_result(job["url"], "applied", duration_ms=duration_ms)
                applied += 1
                update_state(worker_id, jobs_applied=applied,
                             jobs_done=applied + failed)
            else:
                reason = result.split(":", 1)[-1] if ":" in result else result
                mark_result(job["url"], "failed", reason,
                            permanent=_is_permanent_failure(result),
                            duration_ms=duration_ms)
                failed += 1
                update_state(worker_id, jobs_failed=failed,
                             jobs_done=applied + failed)

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
         agent_backend: str = "claude") -> None:
    """Launch the apply pipeline.

    Args:
        limit: Max jobs to apply to (0 or with continuous=True means run forever).
        target_url: Apply to a specific URL.
        min_score: Minimum fit_score threshold.
        headless: Run Chrome in headless mode.
        model: Claude model name.
        dry_run: Don't click Submit.
        continuous: Run forever, polling for new jobs.
        poll_interval: Seconds between DB polls when queue is empty.
        workers: Number of parallel workers (default 1).
        agent_backend: Apply agent backend (claude, copilot, auto).
    """
    global POLL_INTERVAL
    POLL_INTERVAL = poll_interval
    _stop_event.clear()

    config.ensure_dirs()
    console = Console(legacy_windows=False)

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
