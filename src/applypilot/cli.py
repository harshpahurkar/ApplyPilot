"""ApplyPilot CLI — the main entry point."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from applypilot import __version__
from applypilot.config import LOG_DIR

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# File handler — one log file per session
_log_file = LOG_DIR / f"applypilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),   # console (existing behaviour)
        _file_handler,             # file (new)
    ],
)

app = typer.Typer(
    name="applypilot",
    help="AI-powered end-to-end job application pipeline.",
    no_args_is_help=True,
)
console = Console(legacy_windows=False)
log = logging.getLogger(__name__)

# Valid pipeline stages (in execution order)
VALID_STAGES = ("discover", "enrich", "score", "tailor", "cover", "pdf", "resolve_urls")
VALID_AGENT_BACKENDS = ("claude", "copilot", "auto")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Common setup: load env, create dirs, init DB."""
    from applypilot.config import load_env, ensure_dirs
    from applypilot.database import init_db

    load_env()
    ensure_dirs()
    init_db()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]applypilot[/bold] {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """ApplyPilot — AI-powered end-to-end job application pipeline."""


@app.command()
def init() -> None:
    """Run the first-time setup wizard (profile, resume, search config)."""
    from applypilot.wizard.init import run_wizard

    run_wizard()


@app.command()
def run(
    stages: Optional[list[str]] = typer.Argument(
        None,
        help=(
            "Pipeline stages to run. "
            f"Valid: {', '.join(VALID_STAGES)}, all. "
            "Defaults to 'all' if omitted."
        ),
    ),
    min_score: int = typer.Option(6, "--min-score", help="Minimum fit score for tailor/cover stages."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery/enrichment stages."),
    stream: bool = typer.Option(False, "--stream", help="Run stages concurrently (streaming mode)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview stages without executing."),
    validation: str = typer.Option(
        "normal",
        "--validation",
        help=(
            "Validation strictness for tailor/cover stages. "
            "strict: banned words = errors, judge must pass. "
            "normal: banned words = warnings only (default, recommended for Gemini free tier). "
            "lenient: banned words ignored, LLM judge skipped (fastest, fewest API calls)."
        ),
    ),
    stealth: bool = typer.Option(True, "--stealth/--no-stealth", help="Enable human-like pacing for Copilot LLM calls (default: on)."),
) -> None:
    """Run pipeline stages: discover, enrich, score, tailor, cover, pdf, resolve_urls."""
    import os
    os.environ["APPLYPILOT_STEALTH_ENABLED"] = "1" if stealth else "0"

    _bootstrap()

    from applypilot.pipeline import run_pipeline

    stage_list = stages if stages else ["all"]

    # Validate stage names
    for s in stage_list:
        if s != "all" and s not in VALID_STAGES:
            console.print(
                f"[red]Unknown stage:[/red] '{s}'. "
                f"Valid stages: {', '.join(VALID_STAGES)}, all"
            )
            raise typer.Exit(code=1)

    # Gate AI stages behind Tier 2
    llm_stages = {"score", "tailor", "cover"}
    if any(s in stage_list for s in llm_stages) or "all" in stage_list:
        from applypilot.config import check_tier
        check_tier(2, "AI scoring/tailoring")

    # Validate the --validation flag value
    valid_modes = ("strict", "normal", "lenient")
    if validation not in valid_modes:
        console.print(
            f"[red]Invalid --validation value:[/red] '{validation}'. "
            f"Choose from: {', '.join(valid_modes)}"
        )
        raise typer.Exit(code=1)

    result = run_pipeline(
        stages=stage_list,
        min_score=min_score,
        dry_run=dry_run,
        stream=stream,
        workers=workers,
        validation_mode=validation,
    )

    if result.get("errors"):
        raise typer.Exit(code=1)


@app.command()
def apply(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max applications to submit."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel browser workers."),
    min_score: int = typer.Option(6, "--min-score", help="Minimum fit score for job selection."),
    model: str = typer.Option("haiku", "--model", "-m", help="Agent model name."),
    agent_backend: str = typer.Option(
        "claude",
        "--agent-backend",
        help="Agent backend for auto-apply: claude, copilot, or auto.",
    ),
    continuous: bool = typer.Option(False, "--continuous", "-c", help="Run forever, polling for new jobs."),
    poll_interval: int = typer.Option(15, "--poll-interval", help="Seconds between queue polls when no job is available."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without submitting."),
    headless: bool = typer.Option(False, "--headless", help="Run browsers in headless mode."),
    url: Optional[str] = typer.Option(None, "--url", help="Apply to a specific job URL."),
    gen: bool = typer.Option(False, "--gen", help="Generate prompt file for manual debugging instead of running."),
    mark_applied: Optional[str] = typer.Option(None, "--mark-applied", help="Manually mark a job URL as applied."),
    mark_failed: Optional[str] = typer.Option(None, "--mark-failed", help="Manually mark a job URL as failed (provide URL)."),
    fail_reason: Optional[str] = typer.Option(None, "--fail-reason", help="Reason for --mark-failed."),
    reset_failed: bool = typer.Option(False, "--reset-failed", help="Reset all failed jobs for retry."),
    persistent: bool = typer.Option(False, "--persistent", help="Retry transient failures with exponential backoff."),
    stealth: bool = typer.Option(True, "--stealth/--no-stealth", help="Enable human-like pacing and anti-rate-limit evasion (default: on)."),
) -> None:
    """Launch auto-apply to submit job applications."""
    _bootstrap()

    # Configure stealth mode via environment before importing launcher
    import os
    os.environ["APPLYPILOT_STEALTH_ENABLED"] = "1" if stealth else "0"

    from applypilot.config import check_tier, PROFILE_PATH as _profile_path
    from applypilot.database import get_connection

    # --- Utility modes (no Chrome/Claude needed) ---

    if mark_applied:
        from applypilot.apply.launcher import mark_job
        mark_job(mark_applied, "applied")
        console.print(f"[green]Marked as applied:[/green] {mark_applied}")
        return

    if mark_failed:
        from applypilot.apply.launcher import mark_job
        mark_job(mark_failed, "failed", reason=fail_reason)
        console.print(f"[yellow]Marked as failed:[/yellow] {mark_failed} ({fail_reason or 'manual'})")
        return

    if reset_failed:
        from applypilot.apply.launcher import reset_failed as do_reset
        count = do_reset()
        console.print(f"[green]Reset {count} failed job(s) for retry.[/green]")
        return

    # --- Full apply mode ---

    if agent_backend not in VALID_AGENT_BACKENDS:
        console.print(
            f"[red]Invalid --agent-backend value:[/red] '{agent_backend}'. "
            f"Choose from: {', '.join(VALID_AGENT_BACKENDS)}"
        )
        raise typer.Exit(code=1)

    # Check 1: Tier 3 required (Chrome + selected agent backend)
    check_tier(3, "auto-apply", apply_backend=agent_backend)

    # Check 2: Profile exists
    if not _profile_path.exists():
        console.print(
            "[red]Profile not found.[/red]\n"
            "Run [bold]applypilot init[/bold] to create your profile first."
        )
        raise typer.Exit(code=1)

    # Check 3: Tailored resumes exist (skip for --gen with --url)
    if not (gen and url):
        conn = get_connection()
        ready = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND applied_at IS NULL"
        ).fetchone()[0]
        if ready == 0:
            if continuous:
                console.print(
                    "[yellow]No tailored resumes ready yet.[/yellow]\n"
                    "Starting in continuous mode anyway; apply workers will poll until jobs are ready."
                )
            else:
                console.print(
                    "[red]No tailored resumes ready.[/red]\n"
                    "Run [bold]applypilot run score tailor[/bold] first to prepare applications."
                )
                raise typer.Exit(code=1)

    if gen:
        from applypilot.apply.launcher import gen_prompt
        target = url or ""
        if not target:
            console.print("[red]--gen requires --url to specify which job.[/red]")
            raise typer.Exit(code=1)
        prompt_file = gen_prompt(target, min_score=min_score, model=model)
        if not prompt_file:
            console.print("[red]No matching job found for that URL.[/red]")
            raise typer.Exit(code=1)
        mcp_path = _profile_path.parent / ".mcp-apply-0.json"
        console.print(f"[green]Wrote prompt to:[/green] {prompt_file}")
        console.print(f"\n[bold]Run manually:[/bold]")
        if agent_backend == "copilot":
            console.print(
                f"  copilot --model {model} --additional-mcp-config @{mcp_path} "
                f"--allow-all-tools --allow-all-paths --allow-all-urls "
                f"--no-ask-user --output-format json -p \"<paste prompt text from {prompt_file}>\""
            )
        elif agent_backend == "auto":
            console.print(
                f"  claude --model {model} -p "
                f"--mcp-config {mcp_path} "
                f"--permission-mode bypassPermissions < {prompt_file}"
            )
            console.print(
                f"  copilot --model {model} --additional-mcp-config @{mcp_path} "
                f"--allow-all-tools --allow-all-paths --allow-all-urls "
                f"--no-ask-user --output-format json -p \"<paste prompt text from {prompt_file}>\""
            )
        else:
            console.print(
                f"  claude --model {model} -p "
                f"--mcp-config {mcp_path} "
                f"--permission-mode bypassPermissions < {prompt_file}"
            )
        return

    from applypilot.apply.launcher import main as apply_main

    effective_limit = limit if limit is not None else (0 if continuous else 1)

    console.print("\n[bold blue]Launching Auto-Apply[/bold blue]")
    console.print(f"  Limit:    {'unlimited' if continuous else effective_limit}")
    console.print(f"  Workers:  {workers}")
    console.print(f"  Model:    {model}")
    console.print(f"  Backend:  {agent_backend}")
    console.print(f"  Poll:     {poll_interval}s")
    console.print(f"  Headless: {headless}")
    console.print(f"  Dry run:  {dry_run}")
    console.print(f"  Stealth:  {stealth}")
    if url:
        console.print(f"  Target:   {url}")
    console.print()

    apply_main(
        limit=effective_limit,
        target_url=url,
        min_score=min_score,
        headless=headless,
        model=model,
        dry_run=dry_run,
        continuous=continuous,
        poll_interval=poll_interval,
        workers=workers,
        agent_backend=agent_backend,
        persistent=persistent,
    )


@app.command()
def auto(
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery/enrichment."),
    apply_workers: int = typer.Option(1, "--apply-workers", "-a", help="Parallel browser workers for apply."),
    min_score: int = typer.Option(6, "--min-score", help="Minimum fit score."),
    model: str = typer.Option("haiku", "--model", "-m", help="Agent model name for apply."),
    agent_backend: str = typer.Option(
        "claude",
        "--agent-backend",
        help="Agent backend for apply stage: claude, copilot, or auto.",
    ),
    headless: bool = typer.Option(False, "--headless", help="Run browsers in headless mode."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Apply fills forms but does not submit."),
    threshold: int = typer.Option(30, "--threshold", "-t", help="Start applying after this many jobs are ready."),
    stealth: bool = typer.Option(True, "--stealth/--no-stealth", help="Enable human-like pacing and anti-rate-limit evasion (default: on)."),
) -> None:
    """Run full pipeline + auto-apply concurrently (the big one).

    Discovers, enriches, scores, tailors, writes cover letters, converts to PDF,
    and starts applying — ALL AT THE SAME TIME.  The database is the conveyor
    belt: each stage processes jobs as they arrive from upstream.

    Apply workers kick in once --threshold jobs are ready (tailored + PDF'd).
    After that, everything runs in tandem until you Ctrl+C.
    """
    import time as _time
    import os as _os

    _os.environ["APPLYPILOT_STEALTH_ENABLED"] = "1" if stealth else "0"

    _bootstrap()

    from applypilot.config import check_tier, PROFILE_PATH as _profile_path
    from applypilot.pipeline import start_pipeline_background
    from applypilot.database import get_connection, get_stats

    if agent_backend not in VALID_AGENT_BACKENDS:
        console.print(
            f"[red]Invalid --agent-backend value:[/red] '{agent_backend}'. "
            f"Choose from: {', '.join(VALID_AGENT_BACKENDS)}"
        )
        raise typer.Exit(code=1)

    # Must have tier 3 (selected agent CLI + Chrome) and tier 2 (LLM key)
    check_tier(3, "full auto mode", apply_backend=agent_backend)

    if not _profile_path.exists():
        console.print(
            "[red]Profile not found.[/red]\n"
            "Run [bold]applypilot init[/bold] to create your profile first."
        )
        raise typer.Exit(code=1)

    # Banner
    console.print("\n[bold magenta]** FULL AUTO MODE **[/bold magenta]")
    console.print(f"  Pipeline workers: {workers}")
    console.print(f"  Apply workers:    {apply_workers}")
    console.print(f"  Min score:        {min_score}")
    console.print(f"  Model:            {model}")
    console.print(f"  Agent backend:    {agent_backend}")
    console.print(f"  Apply threshold:  {threshold} ready jobs")
    console.print(f"  Headless:         {headless}")
    console.print(f"  Dry run:          {dry_run}")
    console.print(f"  Stealth:          {stealth}")

    # Pre-run stats
    pre = get_stats()
    console.print(f"  DB now:           {pre['total']} jobs, {pre['ready_to_apply']} ready")
    console.print()

    # ── 1. Launch pipeline in background ──────────────────────────────────
    console.print("[cyan]Starting pipeline stages (streaming mode)...[/cyan]")
    stop_event, threads, tracker = start_pipeline_background(
        min_score=min_score,
        workers=workers,
    )
    console.print(f"  [dim]{len(threads)} stage threads launched[/dim]")

    # ── 2. Wait for threshold ─────────────────────────────────────────────
    console.print(f"\n[yellow]Waiting for {threshold} ready-to-apply jobs...[/yellow]")
    console.print("[dim]Ctrl+C to stop everything[/dim]\n")

    try:
        while True:
            conn = get_connection()
            ready_rows = conn.execute(
                "SELECT tailored_resume_path FROM jobs "
                "WHERE tailored_resume_path IS NOT NULL "
                "  AND (apply_status IS NULL OR apply_status IN ('ready', 'failed')) "
                "  AND (apply_attempts IS NULL OR apply_attempts < 5) "
                f"  AND fit_score >= {min_score}"
            ).fetchall()

            ready = 0
            for row in ready_rows:
                resume_path = row["tailored_resume_path"] or ""
                resume_pdf = Path(resume_path).with_suffix(".pdf") if resume_path else None
                if resume_pdf and resume_pdf.exists():
                    ready += 1

            total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            tailored = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL"
            ).fetchone()[0]

            pipeline_alive = any(t.is_alive() for t in threads)

            console.print(
                f"  [dim]{total} discovered · {tailored} tailored · "
                f"{ready} ready to apply · "
                f"{'pipeline running' if pipeline_alive else 'pipeline done'}[/dim]"
            )

            if ready >= threshold:
                console.print(
                    f"\n[green]✓ Threshold reached ({ready} ≥ {threshold}) "
                    f"— launching apply![/green]\n"
                )
                break

            if not pipeline_alive and ready > 0:
                console.print(
                    f"\n[green]✓ Pipeline finished with {ready} ready jobs "
                    f"— launching apply![/green]\n"
                )
                break

            if not pipeline_alive and ready == 0:
                console.print(
                    "\n[red]Pipeline finished but no jobs are ready to apply.[/red]\n"
                    "Run [bold]applypilot status[/bold] to diagnose."
                )
                return

            _time.sleep(10)

        # ── 3. Launch apply (takes over the terminal with dashboard) ──────
        from applypilot.apply.launcher import main as apply_main

        apply_main(
            limit=0,
            min_score=min_score,
            headless=headless,
            model=model,
            dry_run=dry_run,
            continuous=True,
            workers=apply_workers,
            agent_backend=agent_backend,
            persistent=True,
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
    finally:
        stop_event.set()
        alive = [t for t in threads if t.is_alive()]
        if alive:
            console.print(f"[dim]Waiting for {len(alive)} pipeline thread(s)...[/dim]")
            for t in alive:
                t.join(timeout=5)
        console.print("[dim]Done.[/dim]")


@app.command()
def status() -> None:
    """Show pipeline statistics from the database."""
    _bootstrap()

    from applypilot.database import get_stats

    stats = get_stats()

    console.print("\n[bold]ApplyPilot Pipeline Status[/bold]\n")

    # Summary table
    summary = Table(title="Pipeline Overview", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")

    summary.add_row("Total jobs discovered", str(stats["total"]))
    summary.add_row("With full description", str(stats["with_description"]))
    summary.add_row("Pending enrichment", str(stats["pending_detail"]))
    summary.add_row("Enrichment errors", str(stats["detail_errors"]))
    summary.add_row("Scored by LLM", str(stats["scored"]))
    summary.add_row("Pending scoring", str(stats["unscored"]))
    summary.add_row("Tailored resumes", str(stats["tailored"]))
    summary.add_row("Pending tailoring (7+)", str(stats["untailored_eligible"]))
    summary.add_row("Cover letters", str(stats["with_cover_letter"]))
    summary.add_row("Ready to apply", str(stats["ready_to_apply"]))
    summary.add_row("Applied", str(stats["applied"]))
    summary.add_row("Apply errors", str(stats["apply_errors"]))

    console.print(summary)

    # Score distribution
    if stats["score_distribution"]:
        dist_table = Table(title="\nScore Distribution", show_header=True, header_style="bold yellow")
        dist_table.add_column("Score", justify="center")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Bar")

        max_count = max(count for _, count in stats["score_distribution"]) or 1
        for score, count in stats["score_distribution"]:
            bar_len = int(count / max_count * 30)
            if score >= 7:
                color = "green"
            elif score >= 5:
                color = "yellow"
            else:
                color = "red"
            bar = f"[{color}]{'=' * bar_len}[/{color}]"
            dist_table.add_row(str(score), str(count), bar)

        console.print(dist_table)

    # By site
    if stats["by_site"]:
        site_table = Table(title="\nJobs by Source", show_header=True, header_style="bold magenta")
        site_table.add_column("Site")
        site_table.add_column("Count", justify="right")

        for site, count in stats["by_site"]:
            site_table.add_row(site or "Unknown", str(count))

        console.print(site_table)

    console.print()


@app.command()
def dashboard() -> None:
    """Generate and open the HTML dashboard in your browser."""
    _bootstrap()

    from applypilot.view import open_dashboard

    open_dashboard()


@app.command()
def doctor() -> None:
    """Check your setup and diagnose missing requirements."""
    import shutil
    from applypilot.config import (
        load_env, PROFILE_PATH, RESUME_PATH, RESUME_PDF_PATH,
        SEARCH_CONFIG_PATH, ENV_PATH, get_chrome_path,
    )

    load_env()

    ok_mark = "[green]OK[/green]"
    fail_mark = "[red]MISSING[/red]"
    warn_mark = "[yellow]WARN[/yellow]"

    results: list[tuple[str, str, str]] = []  # (check, status, note)

    # --- Tier 1 checks ---
    if PROFILE_PATH.exists():
        results.append(("profile.json", ok_mark, str(PROFILE_PATH)))
    else:
        results.append(("profile.json", fail_mark, "Run 'applypilot init' to create"))

    if RESUME_PATH.exists():
        results.append(("resume.txt", ok_mark, str(RESUME_PATH)))
    elif RESUME_PDF_PATH.exists():
        results.append(("resume.txt", warn_mark, "Only PDF found — plain-text needed for AI stages"))
    else:
        results.append(("resume.txt", fail_mark, "Run 'applypilot init' to add your resume"))

    if SEARCH_CONFIG_PATH.exists():
        results.append(("searches.yaml", ok_mark, str(SEARCH_CONFIG_PATH)))
    else:
        results.append(("searches.yaml", warn_mark, "Will use example config — run 'applypilot init'"))

    try:
        import jobspy  # noqa: F401
        results.append(("python-jobspy", ok_mark, "Job board scraping available"))
    except ImportError:
        results.append(("python-jobspy", warn_mark,
                        "pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex"))

    # --- Tier 2 checks ---
    import os
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_local = bool(os.environ.get("LLM_URL"))
    if has_gemini:
        model = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        results.append(("LLM API key", ok_mark, f"Gemini ({model})"))
    elif has_openai:
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        results.append(("LLM API key", ok_mark, f"OpenAI ({model})"))
    elif has_local:
        results.append(("LLM API key", ok_mark, f"Local: {os.environ.get('LLM_URL')}"))
    else:
        results.append(("LLM API key", fail_mark,
                        "Set GEMINI_API_KEY in ~/.applypilot/.env (run 'applypilot init')"))

    # --- Tier 3 checks ---
    claude_bin = shutil.which("claude")
    if claude_bin:
        results.append(("Claude Code CLI", ok_mark, claude_bin))
    else:
        results.append(("Claude Code CLI", fail_mark,
                        "Install from https://claude.ai/code (one auto-apply backend)"))

    copilot_bin = shutil.which("copilot")
    gh_bin = shutil.which("gh")
    if copilot_bin:
        results.append(("Copilot CLI", ok_mark, copilot_bin))
    elif gh_bin:
        results.append(("Copilot CLI", warn_mark,
                        f"GitHub CLI found at {gh_bin} (install Copilot CLI for agent backend)"))
    else:
        results.append(("Copilot CLI", fail_mark,
                        "Install GitHub Copilot CLI (optional alternative backend)"))

    try:
        chrome_path = get_chrome_path()
        results.append(("Chrome/Chromium", ok_mark, chrome_path))
    except FileNotFoundError:
        results.append(("Chrome/Chromium", fail_mark,
                        "Install Chrome or set CHROME_PATH env var (needed for auto-apply)"))

    npx_bin = shutil.which("npx")
    if npx_bin:
        results.append(("Node.js (npx)", ok_mark, npx_bin))
    else:
        results.append(("Node.js (npx)", fail_mark,
                        "Install Node.js 18+ from nodejs.org (needed for auto-apply)"))

    capsolver = os.environ.get("CAPSOLVER_API_KEY")
    if capsolver:
        results.append(("CapSolver API key", ok_mark, "CAPTCHA solving enabled"))
    else:
        results.append(("CapSolver API key", "[dim]optional[/dim]",
                        "Set CAPSOLVER_API_KEY in .env for CAPTCHA solving"))

    # --- Render results ---
    console.print()
    console.print("[bold]ApplyPilot Doctor[/bold]\n")

    col_w = max(len(r[0]) for r in results) + 2
    for check, status, note in results:
        pad = " " * (col_w - len(check))
        console.print(f"  {check}{pad}{status}  [dim]{note}[/dim]")

    console.print()

    from applypilot.config import get_tier, TIER_LABELS
    tier = get_tier(apply_backend="auto")
    console.print(f"[bold]Current tier: Tier {tier} — {TIER_LABELS[tier]}[/bold]")

    if tier == 1:
        console.print("[dim]  → Tier 2 unlocks: scoring, tailoring, cover letters (needs LLM API key)[/dim]")
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Claude or Copilot CLI + Chrome + Node.js)[/dim]")
    elif tier == 2:
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Claude or Copilot CLI + Chrome + Node.js)[/dim]")

    console.print()


if __name__ == "__main__":
    app()
