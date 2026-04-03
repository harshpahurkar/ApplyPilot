"""ATS Board scrapers: Greenhouse, Lever, Ashby.

Scrapes company career pages on popular ATS platforms via their public
JSON APIs. Zero LLM, zero browser — pure HTTP.

Company registries are loaded from config/ats_boards.yaml.
Title filtering uses the user's search queries from searches.yaml.

Similar architecture to workday.py:
  - Load company registry from YAML
  - For each company: hit API → filter by title keywords → filter by location → store
  - Supports parallel execution via ThreadPoolExecutor
"""

import json
import logging
import sqlite3
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import yaml
from bs4 import BeautifulSoup

from applypilot import config
from applypilot.config import CONFIG_DIR
from applypilot.database import get_connection, init_db

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Default delay between API calls to avoid rate limits (seconds)
REQUEST_DELAY = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    """Convert HTML to plain text."""
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)


def _urlopen(url: str, timeout: int = 30) -> bytes:
    """Fetch a URL and return the raw bytes."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", UA)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_ats_boards() -> dict:
    """Load ATS board registries from config/ats_boards.yaml."""
    path = CONFIG_DIR / "ats_boards.yaml"
    if not path.exists():
        log.warning("ats_boards.yaml not found at %s", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


# ---------------------------------------------------------------------------
# Location filtering (same pattern as workday.py / jobspy.py)
# ---------------------------------------------------------------------------

def _load_location_filter(search_cfg: dict | None = None):
    """Load location accept/reject lists from search config."""
    if search_cfg is None:
        search_cfg = config.load_search_config()
    accept = search_cfg.get("location_accept", [])
    reject = search_cfg.get("location_reject_non_remote", [])
    return accept, reject


def _location_ok(location: str | None, accept: list[str], reject: list[str]) -> bool:
    """Check if a job location passes the user's location filter."""
    return config.location_matches_filter(location, accept, reject, allow_unknown=True)


# ---------------------------------------------------------------------------
# Title keyword matching
# ---------------------------------------------------------------------------

def _build_keywords(queries: list[str]) -> list[str]:
    """Build keyword list from search queries for title matching.

    Combines the full query strings with individual engineering terms
    to catch variations like "Staff Platform Engineer" even if
    "platform engineer" isn't an explicit query.
    """
    keywords = set()
    for q in queries:
        keywords.add(q.lower().replace("-", " "))

    # Core engineering terms for broader matching
    core = [
        "engineer", "developer", "devops", "sre", "architect",
        "programmer", "infrastructure", "platform",
    ]
    keywords.update(core)
    return list(keywords)


def _title_matches(title: str, keywords: list[str]) -> bool:
    """Check if a job title matches any keyword."""
    t = title.lower().replace("-", " ").replace("–", " ")
    return any(k in t for k in keywords)


# ---------------------------------------------------------------------------
# Platform fetchers
# ---------------------------------------------------------------------------

def _fetch_greenhouse(board_token: str, name: str) -> list[dict]:
    """Fetch all jobs from a Greenhouse job board.

    API: GET https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs
    Optional: ?content=true includes full HTML descriptions.
    """
    url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true"
    data = json.loads(_urlopen(url))

    jobs = []
    for j in data.get("jobs", []):
        loc = j.get("location", {})
        loc_name = loc.get("name", "") if isinstance(loc, dict) else str(loc)
        desc_html = j.get("content", "")
        description = _strip_html(desc_html)

        jobs.append({
            "title": j.get("title", ""),
            "location": loc_name,
            "url": j.get("absolute_url", ""),
            "description": description,
            "company": name,
        })

    return jobs


def _fetch_lever(company_slug: str, name: str) -> list[dict]:
    """Fetch all jobs from a Lever job board.

    API: GET https://api.lever.co/v0/postings/{company_slug}
    Returns a JSON array of postings.
    """
    url = f"https://api.lever.co/v0/postings/{company_slug}"
    data = json.loads(_urlopen(url))

    if not isinstance(data, list):
        return []

    jobs = []
    for j in data:
        categories = j.get("categories", {})
        desc = j.get("descriptionPlain", "") or _strip_html(j.get("description", ""))

        jobs.append({
            "title": j.get("text", ""),
            "location": categories.get("location", ""),
            "url": j.get("hostedUrl", ""),
            "description": desc,
            "company": name,
        })

    return jobs


def _fetch_ashby(board_slug: str, name: str) -> list[dict]:
    """Fetch all jobs from an Ashby job board.

    API: GET https://api.ashbyhq.com/posting-api/job-board/{board_slug}
    Returns {"jobs": [...]}.
    """
    url = f"https://api.ashbyhq.com/posting-api/job-board/{board_slug}"
    data = json.loads(_urlopen(url))

    jobs = []
    for j in data.get("jobs", []):
        desc = _strip_html(j.get("descriptionHtml", "")) or j.get("descriptionPlain", "")
        loc = j.get("location", "")
        if isinstance(loc, dict):
            loc = loc.get("name", "")

        # Prefer applicationUrl, fall back to constructed URL
        job_url = j.get("applicationUrl") or j.get("applyUrl") or \
                  f"https://jobs.ashbyhq.com/{board_slug}/{j.get('id', '')}"

        jobs.append({
            "title": j.get("title", ""),
            "location": str(loc),
            "url": job_url,
            "description": desc,
            "company": name,
        })

    return jobs


# Platform dispatcher
_FETCHERS = {
    "greenhouse": lambda c: _fetch_greenhouse(c["board_token"], c["name"]),
    "lever":      lambda c: _fetch_lever(c["company_slug"], c["name"]),
    "ashby":      lambda c: _fetch_ashby(c["board_slug"], c["name"]),
}


# ---------------------------------------------------------------------------
# DB storage
# ---------------------------------------------------------------------------

def _store_results(
    conn: sqlite3.Connection,
    jobs: list[dict],
    site: str,
    strategy: str,
    accept_locs: list[str],
    reject_locs: list[str],
) -> tuple[int, int]:
    """Store jobs with location filtering. Returns (new, existing)."""
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    existing = 0
    filtered = 0

    for job in jobs:
        url = job.get("url")
        if not url:
            continue

        if not _location_ok(job.get("location"), accept_locs, reject_locs):
            filtered += 1
            continue

        description = job.get("description", "")
        short_desc = description[:500] if description else None
        full_desc = description if len(description) > 200 else None
        detail_scraped_at = now if full_desc else None

        try:
            conn.execute(
                "INSERT INTO jobs (url, title, salary, description, location, site, strategy, "
                "discovered_at, full_description, application_url, detail_scraped_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (url, job.get("title"), None, short_desc, job.get("location"),
                 site, strategy, now, full_desc, url, detail_scraped_at),
            )
            new += 1
        except sqlite3.IntegrityError:
            existing += 1

    if filtered:
        log.info("  Filtered %d jobs (wrong location)", filtered)
    conn.commit()
    return new, existing


# ---------------------------------------------------------------------------
# Per-company scraper
# ---------------------------------------------------------------------------

def _scrape_company(
    platform: str,
    key: str,
    company: dict,
    keywords: list[str],
    accept_locs: list[str],
    reject_locs: list[str],
) -> dict:
    """Scrape one company from an ATS platform.

    Returns a result dict with: company, platform, total, matched, new, existing, error (if any).
    """
    name = company.get("name", key)
    fetcher = _FETCHERS.get(platform)
    if not fetcher:
        return {"company": name, "platform": platform, "total": 0, "matched": 0,
                "new": 0, "existing": 0, "error": f"unknown platform: {platform}"}

    try:
        all_jobs = fetcher(company)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            log.debug("%s/%s: board not found (404) — check board token", platform, name)
        else:
            log.warning("%s/%s: HTTP %d", platform, name, e.code)
        return {"company": name, "platform": platform, "total": 0, "matched": 0,
                "new": 0, "existing": 0, "error": f"HTTP {e.code}"}
    except Exception as e:
        log.warning("%s/%s: %s", platform, name, e)
        return {"company": name, "platform": platform, "total": 0, "matched": 0,
                "new": 0, "existing": 0, "error": str(e)[:100]}

    total = len(all_jobs)

    # Filter by title keywords
    matched = [j for j in all_jobs if _title_matches(j.get("title", ""), keywords)]

    if not matched:
        log.debug("%s/%s: %d total, 0 matched keywords", platform, name, total)
        return {"company": name, "platform": platform, "total": total, "matched": 0,
                "new": 0, "existing": 0}

    # Store with location filtering
    conn = get_connection()
    new, existing = _store_results(
        conn, matched, name, f"{platform}_api", accept_locs, reject_locs,
    )

    log.info("%s/%s: %d total → %d keyword match → +%d new, %d dupes",
             platform, name, total, len(matched), new, existing)

    return {"company": name, "platform": platform, "total": total,
            "matched": len(matched), "new": new, "existing": existing}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_ats_discovery(workers: int = 1) -> dict:
    """Main entry point for ATS board job discovery.

    Loads company registries from config/ats_boards.yaml and search queries
    from the user's search config, then scrapes all configured ATS boards.

    Args:
        workers: Number of parallel threads. Default 1 (sequential).

    Returns:
        Dict with stats: found, new, existing, errors, companies_scraped.
    """
    boards = load_ats_boards()
    if not boards:
        log.warning("No ATS boards configured. Create config/ats_boards.yaml.")
        return {"found": 0, "new": 0, "existing": 0, "errors": 0, "companies_scraped": 0}

    # Load search config
    search_cfg = config.load_search_config()
    queries_cfg = search_cfg.get("queries", [])
    max_tier = search_cfg.get("ats_max_tier", 2)
    queries = [q["query"] for q in queries_cfg if q.get("tier", 99) <= max_tier]
    if not queries:
        queries = [q["query"] for q in queries_cfg]

    accept_locs, reject_locs = _load_location_filter(search_cfg)
    keywords = _build_keywords(queries)

    # Ensure DB schema
    init_db()

    grand_new = 0
    grand_existing = 0
    grand_found = 0
    grand_errors = 0
    companies_scraped = 0

    total_companies = sum(
        len(boards.get(p, {})) for p in ["greenhouse", "lever", "ashby"]
    )
    log.info("ATS boards: %d companies across Greenhouse/Lever/Ashby (workers=%d)",
             total_companies, workers)

    t0 = time.time()

    for platform in ["greenhouse", "lever", "ashby"]:
        companies = boards.get(platform, {})
        if not companies:
            continue

        log.info("ATS [%s]: scraping %d companies...", platform, len(companies))
        items = list(companies.items())
        platform_t0 = time.time()

        if workers > 1 and len(items) > 1:
            # Parallel mode
            with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
                futures = {
                    pool.submit(
                        _scrape_company, platform, key, company,
                        keywords, accept_locs, reject_locs,
                    ): key
                    for key, company in items
                }
                for future in as_completed(futures):
                    result = future.result()
                    companies_scraped += 1
                    grand_new += result["new"]
                    grand_existing += result["existing"]
                    grand_found += result["matched"]
                    if result.get("error"):
                        grand_errors += 1
        else:
            # Sequential mode (default) with rate limiting
            for key, company in items:
                result = _scrape_company(
                    platform, key, company,
                    keywords, accept_locs, reject_locs,
                )
                companies_scraped += 1
                grand_new += result["new"]
                grand_existing += result["existing"]
                grand_found += result["matched"]
                if result.get("error"):
                    grand_errors += 1
                time.sleep(REQUEST_DELAY)

        elapsed = time.time() - platform_t0
        log.info("ATS [%s]: done in %.0fs", platform, elapsed)

    total_elapsed = time.time() - t0
    log.info("ATS boards done: %d companies scraped, %d matched, %d new, %d existing, %d errors [%.0fs]",
             companies_scraped, grand_found, grand_new, grand_existing, grand_errors, total_elapsed)

    return {
        "found": grand_found,
        "new": grand_new,
        "existing": grand_existing,
        "errors": grand_errors,
        "companies_scraped": companies_scraped,
    }
