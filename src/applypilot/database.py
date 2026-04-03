"""ApplyPilot database layer: schema, migrations, stats, and connection helpers.

Single source of truth for the jobs table schema. All columns from every
pipeline stage are created up front so any stage can run independently
without migration ordering issues.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from applypilot.config import DB_PATH, DEFAULTS, load_blocked_sites

# Thread-local connection storage — each thread gets its own connection
# (required for SQLite thread safety with parallel workers)
_local = threading.local()


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Get a thread-local cached SQLite connection with WAL mode enabled.

    Each thread gets its own connection (required for SQLite thread safety).
    Connections are cached and reused within the same thread.

    Args:
        db_path: Override the default DB_PATH. Useful for testing.

    Returns:
        sqlite3.Connection configured with WAL mode and row factory.
    """
    path = str(db_path or DB_PATH)

    if not hasattr(_local, 'connections'):
        _local.connections = {}

    conn = _local.connections.get(path)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.ProgrammingError:
            pass

    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    _local.connections[path] = conn
    return conn


def close_connection(db_path: Path | str | None = None) -> None:
    """Close the cached connection for the current thread."""
    path = str(db_path or DB_PATH)
    if hasattr(_local, 'connections'):
        conn = _local.connections.pop(path, None)
        if conn is not None:
            conn.close()


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create the full jobs table with all columns from every pipeline stage.

    This is idempotent -- safe to call on every startup. Uses CREATE TABLE IF NOT EXISTS
    so it won't destroy existing data.

    Schema columns by stage:
      - Discovery:  url, title, salary, description, location, site, strategy, discovered_at
      - Enrichment: full_description, application_url, detail_scraped_at, detail_error
      - Scoring:    fit_score, score_reasoning, scored_at
      - Tailoring:  tailored_resume_path, tailored_at, tailor_attempts
      - Cover:      cover_letter_path, cover_letter_at, cover_attempts
      - Apply:      applied_at, apply_status, apply_error, apply_attempts,
                   agent_id, last_attempted_at, apply_duration_ms, apply_task_id,
                   verification_confidence

    Args:
        db_path: Override the default DB_PATH.

    Returns:
        sqlite3.Connection with the schema initialized.
    """
    path = db_path or DB_PATH

    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            -- Discovery stage (smart_extract / job_search)
            url                   TEXT PRIMARY KEY,
            title                 TEXT,
            salary                TEXT,
            description           TEXT,
            location              TEXT,
            site                  TEXT,
            strategy              TEXT,
            discovered_at         TEXT,
            date_posted           TEXT,

            -- Enrichment stage (detail_scraper)
            full_description      TEXT,
            application_url       TEXT,
            detail_scraped_at     TEXT,
            detail_error          TEXT,

            -- Scoring stage (job_scorer)
            fit_score             INTEGER,
            score_reasoning       TEXT,
            scored_at             TEXT,

            -- Tailoring stage (resume tailor)
            tailored_resume_path  TEXT,
            tailored_at           TEXT,
            tailor_attempts       INTEGER DEFAULT 0,

            -- Cover letter stage
            cover_letter_path     TEXT,
            cover_letter_at       TEXT,
            cover_attempts        INTEGER DEFAULT 0,

            -- Application stage
            applied_at            TEXT,
            apply_status          TEXT,
            apply_error           TEXT,
            apply_attempts        INTEGER DEFAULT 0,
            agent_id              TEXT,
            last_attempted_at     TEXT,
            apply_duration_ms     INTEGER,
            apply_task_id         TEXT,
            verification_confidence TEXT
        )
    """)
    conn.commit()

    # Run migrations for any columns added after initial schema
    ensure_columns(conn)

    # Create indexes for common query patterns (idempotent)
    _ensure_indexes(conn)

    # Named migrations for schema changes beyond column additions
    run_migrations(conn)

    # Session tracking table (pipeline run metadata)
    _ensure_sessions_table(conn)

    return conn


# Complete column registry: column_name -> SQL type with optional default.
# This is the single source of truth. Adding a column here is all that's needed
# for it to appear in both new databases and migrated ones.
_ALL_COLUMNS: dict[str, str] = {
    # Discovery
    "url": "TEXT PRIMARY KEY",
    "title": "TEXT",
    "salary": "TEXT",
    "description": "TEXT",
    "location": "TEXT",
    "site": "TEXT",
    "strategy": "TEXT",
    "discovered_at": "TEXT",
    "date_posted": "TEXT",
    # Enrichment
    "full_description": "TEXT",
    "application_url": "TEXT",
    "detail_scraped_at": "TEXT",
    "detail_error": "TEXT",
    # Scoring
    "fit_score": "INTEGER",
    "score_reasoning": "TEXT",
    "scored_at": "TEXT",
    # Tailoring
    "tailored_resume_path": "TEXT",
    "tailored_at": "TEXT",
    "tailor_attempts": "INTEGER DEFAULT 0",
    # Cover letter
    "cover_letter_path": "TEXT",
    "cover_letter_at": "TEXT",
    "cover_attempts": "INTEGER DEFAULT 0",
    # Application
    "applied_at": "TEXT",
    "apply_status": "TEXT",
    "apply_error": "TEXT",
    "apply_attempts": "INTEGER DEFAULT 0",
    "agent_id": "TEXT",
    "last_attempted_at": "TEXT",
    "apply_duration_ms": "INTEGER",
    "apply_task_id": "TEXT",
    "verification_confidence": "TEXT",
}


def ensure_columns(conn: sqlite3.Connection | None = None) -> list[str]:
    """Add any missing columns to the jobs table (forward migration).

    Reads the current table schema via PRAGMA table_info and compares against
    the full column registry. Any missing columns are added with ALTER TABLE.

    This makes it safe to upgrade the database from any previous version --
    columns are only added, never removed or renamed.

    Args:
        conn: Database connection. Uses get_connection() if None.

    Returns:
        List of column names that were added (empty if schema was already current).
    """
    if conn is None:
        conn = get_connection()

    existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    added = []

    for col, dtype in _ALL_COLUMNS.items():
        if col not in existing:
            # PRIMARY KEY columns can't be added via ALTER TABLE, but url
            # is always created with the table itself so this is safe
            if "PRIMARY KEY" in dtype:
                continue
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {dtype}")
            added.append(col)

    if added:
        conn.commit()

    return added


# Indexes for common query patterns — dramatically improves performance
# at scale (1000+ jobs). All are CREATE IF NOT EXISTS so safe to re-run.
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_jobs_apply_status ON jobs(apply_status)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_fit_score ON jobs(fit_score)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_site ON jobs(site)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_discovered_at ON jobs(discovered_at)",
    # Compound index for the most common query: ready-to-apply
    "CREATE INDEX IF NOT EXISTS idx_jobs_apply_ready ON jobs(apply_status, fit_score, site)",
    # Enrichment queue
    "CREATE INDEX IF NOT EXISTS idx_jobs_pending_detail ON jobs(detail_scraped_at)",
    # Scoring queue
    "CREATE INDEX IF NOT EXISTS idx_jobs_pending_score ON jobs(full_description, fit_score)",
]


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    """Create performance indexes on the jobs table (idempotent)."""
    for ddl in _INDEXES:
        conn.execute(ddl)
    conn.commit()


# ---------------------------------------------------------------------------
# Named migration system — for schema changes beyond column additions.
#
# Each migration is a (name, callable) pair.  `ensure_columns()` above still
# handles new columns as a safety net, but index changes, type tweaks, data
# backfills, or new tables should be registered here.
#
# Migrations run in declaration order and are idempotent: a `_migrations`
# table tracks which have been applied so they never re-run.
# ---------------------------------------------------------------------------

def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            name       TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
    """)
    conn.commit()


# Registry — append new migrations at the end.  Each entry is
# ("unique_name", callable(conn)).  The callable MUST be idempotent because
# in edge cases it might run twice (e.g. crash between execute and commit).
_NAMED_MIGRATIONS: list[tuple[str, callable]] = [
    # Example:
    # ("001_add_foo_column", lambda conn: conn.execute(
    #     "ALTER TABLE jobs ADD COLUMN foo TEXT"
    # )),
]


def run_migrations(conn: sqlite3.Connection | None = None) -> list[str]:
    """Run all unapplied named migrations in order.

    Returns list of migration names that were applied this run.
    """
    if conn is None:
        conn = get_connection()

    _ensure_migrations_table(conn)

    applied = {
        row[0]
        for row in conn.execute("SELECT name FROM _migrations").fetchall()
    }

    newly_applied: list[str] = []
    for name, fn in _NAMED_MIGRATIONS:
        if name in applied:
            continue
        try:
            fn(conn)
            conn.execute(
                "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
                (name, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            newly_applied.append(name)
        except Exception:
            conn.rollback()
            raise

    return newly_applied


def get_stats(conn: sqlite3.Connection | None = None) -> dict:
    """Return job counts by pipeline stage.

    Provides a snapshot of how many jobs are at each stage, useful for
    dashboard display and pipeline progress tracking.

    Args:
        conn: Database connection. Uses get_connection() if None.

    Returns:
        Dictionary with keys:
            total, by_site, pending_detail, with_description,
            scored, unscored, tailored, untailored_eligible,
            with_cover_letter, applied, score_distribution
    """
    if conn is None:
        conn = get_connection()

    stats: dict = {}

    # Total jobs
    stats["total"] = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    # By site breakdown
    rows = conn.execute(
        "SELECT site, COUNT(*) as cnt FROM jobs GROUP BY site ORDER BY cnt DESC"
    ).fetchall()
    stats["by_site"] = [(row[0], row[1]) for row in rows]

    # Enrichment stage
    stats["pending_detail"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE detail_scraped_at IS NULL"
    ).fetchone()[0]

    stats["with_description"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE full_description IS NOT NULL"
    ).fetchone()[0]

    stats["detail_errors"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE detail_error IS NOT NULL"
    ).fetchone()[0]

    # Scoring stage
    stats["scored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE fit_score IS NOT NULL"
    ).fetchone()[0]

    stats["unscored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE full_description IS NOT NULL AND fit_score IS NULL"
    ).fetchone()[0]

    # Score distribution
    dist_rows = conn.execute(
        "SELECT fit_score, COUNT(*) as cnt FROM jobs "
        "WHERE fit_score IS NOT NULL "
        "GROUP BY fit_score ORDER BY fit_score DESC"
    ).fetchall()
    stats["score_distribution"] = [(row[0], row[1]) for row in dist_rows]

    # Tailoring stage
    stats["tailored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL"
    ).fetchone()[0]

    stats["untailored_eligible"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE fit_score >= 7 AND full_description IS NOT NULL "
        "AND tailored_resume_path IS NULL"
    ).fetchone()[0]

    stats["tailor_exhausted"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE COALESCE(tailor_attempts, 0) >= 5 "
        "AND tailored_resume_path IS NULL"
    ).fetchone()[0]

    # Cover letter stage
    stats["with_cover_letter"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE cover_letter_path IS NOT NULL"
    ).fetchone()[0]

    stats["cover_exhausted"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE COALESCE(cover_attempts, 0) >= 5 "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '')"
    ).fetchone()[0]

    # Application stage
    stats["applied"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE apply_status = 'applied'"
    ).fetchone()[0]

    stats["apply_errors"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE apply_error IS NOT NULL"
    ).fetchone()[0]

    blocked_sites, blocked_patterns = load_blocked_sites()
    blocked_sites_lc = {s.lower() for s in blocked_sites}
    normalized_patterns = [p.replace("%", "").lower() for p in blocked_patterns if p]

    ready_rows = conn.execute(
        "SELECT site, url, application_url FROM jobs "
        "WHERE tailored_resume_path IS NOT NULL "
        "AND (apply_status IS NULL OR apply_status IN ('ready', 'failed')) "
        "AND (apply_attempts IS NULL OR apply_attempts < ?) "
        "AND application_url IS NOT NULL "
        "AND application_url != ''",
        (DEFAULTS["max_apply_attempts"],),
    ).fetchall()

    ready_effective = 0
    for row in ready_rows:
        site = (row["site"] or "").lower()
        apply_url = (row["application_url"] or row["url"] or "").lower()

        if site in blocked_sites_lc:
            continue
        if any(pattern in apply_url for pattern in normalized_patterns):
            continue
        if "linkedin.com" in apply_url:
            continue

        ready_effective += 1

    stats["ready_to_apply"] = ready_effective

    return stats


def store_jobs(conn: sqlite3.Connection, jobs: list[dict],
               site: str, strategy: str) -> tuple[int, int]:
    """Store discovered jobs, skipping duplicates by URL.

    Args:
        conn: Database connection.
        jobs: List of job dicts with keys: url, title, salary, description, location.
        site: Source site name (e.g. "RemoteOK", "Dice").
        strategy: Extraction strategy used (e.g. "json_ld", "api_response", "css_selectors").

    Returns:
        Tuple of (new_count, duplicate_count).
    """
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    existing = 0

    for job in jobs:
        url = job.get("url")
        if not url:
            continue
        try:
            conn.execute(
                "INSERT INTO jobs (url, title, salary, description, location, site, strategy, discovered_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (url, job.get("title"), job.get("salary"), job.get("description"),
                 job.get("location"), site, strategy, now),
            )
            new += 1
        except sqlite3.IntegrityError:
            existing += 1

    conn.commit()
    return new, existing


def get_jobs_by_stage(conn: sqlite3.Connection | None = None,
                      stage: str = "discovered",
                      min_score: int | None = None,
                      limit: int = 100) -> list[dict]:
    """Fetch jobs filtered by pipeline stage.

    Args:
        conn: Database connection. Uses get_connection() if None.
        stage: One of "discovered", "enriched", "scored", "tailored", "applied".
        min_score: Minimum fit_score filter (only relevant for scored+ stages).
        limit: Maximum number of rows to return.

    Returns:
        List of job dicts.
    """
    if conn is None:
        conn = get_connection()

    conditions = {
        "discovered": "1=1",
        "pending_detail": "detail_scraped_at IS NULL",
        "enriched": "full_description IS NOT NULL",
        "pending_score": "full_description IS NOT NULL AND fit_score IS NULL",
        "scored": "fit_score IS NOT NULL",
        "pending_tailor": (
            "fit_score >= ? AND full_description IS NOT NULL "
            "AND tailored_resume_path IS NULL AND COALESCE(tailor_attempts, 0) < 5"
        ),
        "tailored": "tailored_resume_path IS NOT NULL",
        "pending_apply": (
            "tailored_resume_path IS NOT NULL AND applied_at IS NULL "
            "AND application_url IS NOT NULL"
        ),
        "applied": "applied_at IS NOT NULL",
    }

    where = conditions.get(stage, "1=1")
    params: list = []

    if "?" in where and min_score is not None:
        params.append(min_score)
    elif "?" in where:
        params.append(7)  # default min_score

    if min_score is not None and "fit_score" not in where and stage in ("scored", "tailored", "applied"):
        where += " AND fit_score >= ?"
        params.append(min_score)

    query = f"SELECT * FROM jobs WHERE {where} ORDER BY COALESCE(date_posted, discovered_at) DESC, fit_score DESC NULLS LAST"
    if limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()

    # Convert sqlite3.Row objects to dicts
    if rows:
        columns = rows[0].keys()
        return [dict(zip(columns, row)) for row in rows]
    return []


# ---------------------------------------------------------------------------
# Session tracking — records pipeline run metadata for crash recovery and
# diagnostics. Inspired by Claude Code's session state persistence.
# ---------------------------------------------------------------------------

def _ensure_sessions_table(conn: sqlite3.Connection) -> None:
    """Create the sessions table (idempotent)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id    TEXT PRIMARY KEY,
            started_at    TEXT NOT NULL,
            finished_at   TEXT,
            mode          TEXT,
            stages        TEXT,
            status        TEXT DEFAULT 'running',
            jobs_processed INTEGER DEFAULT 0,
            jobs_applied   INTEGER DEFAULT 0,
            total_cost_usd REAL DEFAULT 0.0,
            error_summary  TEXT,
            extra_json     TEXT
        )
    """)
    conn.commit()


def create_session(
    session_id: str,
    mode: str = "pipeline",
    stages: str = "",
) -> None:
    """Record a new pipeline session start."""
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO sessions "
        "(session_id, started_at, mode, stages, status) "
        "VALUES (?, ?, ?, ?, 'running')",
        (session_id, now, mode, stages),
    )
    conn.commit()


def update_session(
    session_id: str,
    *,
    status: str | None = None,
    jobs_processed: int | None = None,
    jobs_applied: int | None = None,
    total_cost_usd: float | None = None,
    error_summary: str | None = None,
    extra_json: str | None = None,
) -> None:
    """Update an existing session with latest progress."""
    conn = get_connection()
    sets: list[str] = []
    params: list = []

    if status is not None:
        sets.append("status = ?")
        params.append(status)
        if status in ("completed", "failed", "interrupted"):
            sets.append("finished_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())
    if jobs_processed is not None:
        sets.append("jobs_processed = ?")
        params.append(jobs_processed)
    if jobs_applied is not None:
        sets.append("jobs_applied = ?")
        params.append(jobs_applied)
    if total_cost_usd is not None:
        sets.append("total_cost_usd = ?")
        params.append(total_cost_usd)
    if error_summary is not None:
        sets.append("error_summary = ?")
        params.append(error_summary)
    if extra_json is not None:
        sets.append("extra_json = ?")
        params.append(extra_json)

    if not sets:
        return
    params.append(session_id)
    conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE session_id = ?", params)
    conn.commit()


def get_last_session() -> dict | None:
    """Return the most recent session record, or None."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM sessions ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if row:
        return dict(zip(row.keys(), row))
    return None
