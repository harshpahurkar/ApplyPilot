"""URL Resolver: finds application URLs for jobs that are missing them.

For jobs (especially LinkedIn-sourced) where the enrichment stage captured
the full_description but NOT the application_url, this module tries to
find the actual company careers page / ATS application link.

Three-tier resolution cascade:
  Tier 1: Scrape the original listing page for an external apply redirect
  Tier 2: Web search (DuckDuckGo) using company name + job title
  Tier 3: LLM-assisted company extraction → targeted ATS search

Runs as a pipeline stage: 'resolve_urls', between 'pdf' and 'apply'.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from urllib.parse import quote_plus, urljoin

from applypilot.config import DB_PATH
from applypilot.database import get_connection, init_db

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Common ATS domains — if we find URLs matching these, they're high-confidence
ATS_DOMAINS = [
    "greenhouse.io", "boards.greenhouse.io",
    "lever.co", "jobs.lever.co",
    "workday.com", "myworkdayjobs.com",
    "smartrecruiters.com",
    "ashbyhq.com", "jobs.ashbyhq.com",
    "icims.com",
    "taleo.net",
    "jobvite.com",
    "bamboohr.com",
    "jazz.co", "applytojob.com",
    "breezy.hr",
    "recruitee.com",
    "ultipro.com",
    "ceridian.com", "dayforce.com",
    "successfactors.com",
    "apploi.com",
    "paylocity.com",
    "workable.com",
    "careers-page.com",
]

# Patterns that indicate a URL is an actual apply/careers page
APPLY_URL_PATTERNS = [
    r"/jobs?/",
    r"/careers?/",
    r"/positions?/",
    r"/openings?/",
    r"/apply",
    r"/job-board",
    r"/posting",
    r"/requisition",
]


def _is_ats_url(url: str) -> bool:
    """Check if URL belongs to a known ATS platform."""
    url_lower = url.lower()
    return any(domain in url_lower for domain in ATS_DOMAINS)


def _is_careers_url(url: str) -> bool:
    """Check if URL looks like a careers/job application page."""
    url_lower = url.lower()
    return _is_ats_url(url_lower) or any(
        re.search(pat, url_lower) for pat in APPLY_URL_PATTERNS
    )


def _extract_company_from_jd(full_description: str, title: str) -> str | None:
    """Try to extract the company name from the JD text using patterns.

    Looks for common patterns like:
      - "About <Company>"
      - "<Company> is hiring"
      - "at <Company>"
      - "Join <Company>"
    """
    if not full_description:
        return None

    text = full_description[:2000]

    # Pattern: "About <Company>" / "About The Company" section often names it
    patterns = [
        r"(?:about|join|at)\s+([A-Z][A-Za-z0-9\s&\-\.]{2,30}?)(?:\s*[,\.\!\n]|\s+is\b|\s+we\b)",
        r"([A-Z][A-Za-z0-9\s&\-\.]{2,25}?)\s+is\s+(?:a|an|the|one of|looking|hiring|seeking)",
        r"(?:company|employer|organization):\s*([A-Za-z0-9\s&\-\.]{2,30})",
        r"(?:work(?:ing)?\s+(?:at|for))\s+([A-Z][A-Za-z0-9\s&\-\.]{2,30}?)(?:\s*[,\.\!\n])",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            company = m.group(1).strip()
            # Filter out generic words
            skip = {"the role", "the team", "our team", "this role", "us",
                    "the company", "our company", "a company", "an organization"}
            if company.lower() not in skip and len(company) > 2:
                return company

    return None


def _extract_company_with_llm(full_description: str, title: str) -> str | None:
    """Use LLM to extract the company/employer name from the job description."""
    try:
        from applypilot.llm import get_client

        prompt = (
            "Extract ONLY the company/employer name from this job posting. "
            "Return just the company name, nothing else. If you cannot determine "
            "the company name, return 'UNKNOWN'.\n\n"
            f"Job Title: {title}\n\n"
            f"Description (first 1500 chars):\n{full_description[:1500]}"
        )
        client = get_client()
        raw = client.ask(prompt, temperature=0.0, max_tokens=50)
        company = raw.strip().strip('"').strip("'").strip()

        if company and company.upper() != "UNKNOWN" and len(company) < 60:
            return company
    except Exception as e:
        log.warning("LLM company extraction failed: %s", e)

    return None


# ---------------------------------------------------------------------------
# Tier 0: Extract company name from LinkedIn page (no login needed)
# ---------------------------------------------------------------------------

def _scrape_linkedin_company(url: str) -> str | None:
    """Scrape the company name from a LinkedIn job listing page.

    LinkedIn shows company name even without login in the page metadata
    and structured data. This is faster and cheaper than an LLM call.
    """
    try:
        import httpx
        from bs4 import BeautifulSoup

        resp = httpx.get(url, headers={"User-Agent": UA}, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Method 1: og:title often has "Company hiring Title in Location"
        og_title = soup.find("meta", property="og:title")
        if og_title:
            content = og_title.get("content", "")
            # Pattern: "Company hiring Job Title in Location"
            m = re.match(r"^(.+?)\s+hiring\s+", content)
            if m:
                return m.group(1).strip()

        # Method 2: JSON-LD JobPosting
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                if isinstance(data, dict):
                    if data.get("@type") == "JobPosting":
                        org = data.get("hiringOrganization", {})
                        if isinstance(org, dict):
                            name = org.get("name")
                            if name:
                                return name
                    elif "@graph" in data:
                        for item in data["@graph"]:
                            if isinstance(item, dict) and item.get("@type") == "JobPosting":
                                org = item.get("hiringOrganization", {})
                                if isinstance(org, dict):
                                    return org.get("name")
            except (json.JSONDecodeError, TypeError):
                continue

        # Method 3: title tag — often "Job Title - Company | LinkedIn"
        title_tag = soup.find("title")
        if title_tag:
            title_text = title_tag.get_text()
            # "Job Title - Company Name | LinkedIn"
            parts = title_text.split(" | ")
            if len(parts) >= 2:
                pre_pipe = parts[0]
                dash_parts = pre_pipe.rsplit(" - ", 1)
                if len(dash_parts) == 2:
                    return dash_parts[1].strip()

        # Method 4: Look for company name in specific classes
        for sel in [".topcard__org-name-link", ".top-card-layout__company-url",
                    'a[data-tracking-control-name="public_jobs_topcard-org-name"]']:
            tag = soup.select_one(sel)
            if tag:
                name = tag.get_text(strip=True)
                if name:
                    return name

    except Exception as e:
        log.debug("LinkedIn company scrape failed for %s: %s", url[:50], e)

    return None


# ---------------------------------------------------------------------------
# Tier 1: Scrape original listing page for external apply link
# ---------------------------------------------------------------------------

def _resolve_from_listing_page(url: str) -> str | None:
    """Visit the original listing (e.g. LinkedIn) and look for external apply link.

    LinkedIn job pages sometimes show an 'Apply on company website' button
    that redirects to the actual ATS/careers page. We capture that redirect.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.warning("Playwright not available for Tier 1 resolution")
        return None

    apply_url = None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=UA)
            page = context.new_page()

            # Capture any navigation redirects from clicking apply
            external_urls: list[str] = []

            def _on_response(response):
                """Track redirects to external sites."""
                resp_url = response.url
                if resp_url and "linkedin.com" not in resp_url:
                    if _is_careers_url(resp_url):
                        external_urls.append(resp_url)

            page.on("response", _on_response)

            try:
                page.goto(url, timeout=30000)
                page.wait_for_load_state("domcontentloaded", timeout=10000)
            except Exception as e:
                log.debug("Page load failed for %s: %s", url, e)
                browser.close()
                return None

            # Strategy A: Look for "Apply on company website" type links
            selectors = [
                'a[href*="apply"]',
                'a[data-tracking-control-name*="apply"]',
                'a.apply-button',
                'button.jobs-apply-button',
                'a[class*="apply"]',
            ]

            for sel in selectors:
                try:
                    el = page.query_selector(sel)
                    if el:
                        href = el.get_attribute("href")
                        if href and "linkedin.com" not in href and href.startswith("http"):
                            apply_url = href
                            break
                except Exception:
                    continue

            # Strategy B: Check for external URLs captured during page load
            if not apply_url and external_urls:
                # Prefer ATS URLs
                for eu in external_urls:
                    if _is_ats_url(eu):
                        apply_url = eu
                        break
                if not apply_url:
                    apply_url = external_urls[0]

            # Strategy C: Look for company website links
            if not apply_url:
                try:
                    links = page.query_selector_all("a")
                    for link in links:
                        href = link.get_attribute("href") or ""
                        text = (link.inner_text() or "").strip().lower()
                        if (href.startswith("http")
                                and "linkedin.com" not in href
                                and any(kw in text for kw in ("apply", "website", "career"))):
                            apply_url = href
                            break
                except Exception:
                    pass

            browser.close()

    except Exception as e:
        log.warning("Tier 1 (listing scrape) failed for %s: %s", url, e)

    return apply_url


# ---------------------------------------------------------------------------
# Tier 2: Web search for the job posting
# ---------------------------------------------------------------------------

def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return results.

    Returns list of {"title": str, "url": str, "body": str}.
    Uses the ddgs package (preferred) or duckduckgo_search fallback.
    """
    # Try ddgs (current package name)
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [{"title": r.get("title", ""), "url": r.get("href", ""),
                 "body": r.get("body", "")} for r in results]
    except ImportError:
        pass
    except Exception as e:
        log.warning("ddgs package search failed: %s", e)

    # Fallback: old package name
    try:
        from duckduckgo_search import DDGS as DDGS_Old

        with DDGS_Old() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [{"title": r.get("title", ""), "url": r.get("href", ""),
                 "body": r.get("body", "")} for r in results]
    except ImportError:
        pass
    except Exception as e:
        log.warning("duckduckgo_search fallback failed: %s", e)

    # Last resort: HTTP scrape of DuckDuckGo lite
    try:
        import httpx

        resp = httpx.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": UA},
            timeout=15,
            follow_redirects=True,
        )
        resp.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for a_tag in soup.select("a.result-link, td a[href^='http']"):
            href = a_tag.get("href", "")
            title = a_tag.get_text(strip=True)
            if href.startswith("http") and "duckduckgo" not in href:
                results.append({"title": title, "url": href, "body": ""})
            if len(results) >= max_results:
                break

        return results
    except Exception as e:
        log.warning("DuckDuckGo lite fallback failed: %s", e)

    return []


def _resolve_via_search(title: str, company: str | None,
                         full_description: str | None) -> str | None:
    """Search the web for the actual job application page.

    Builds a targeted search query and looks for ATS/careers URLs.
    """
    # Build search query
    parts = []
    if company:
        parts.append(f'"{company}"')
    parts.append(f'"{title}"')
    parts.append("apply careers")

    query = " ".join(parts)
    log.info("Tier 2 search: %s", query)

    results = _search_duckduckgo(query, max_results=10)

    if not results:
        # Try a simpler query
        simple_query = f"{company or ''} {title} job apply"
        results = _search_duckduckgo(simple_query.strip(), max_results=10)

    # If still no results and we have a company, try company careers directly
    if not results and company:
        results = _search_duckduckgo(f"{company} careers jobs", max_results=10)

    # Score results: prefer ATS domains, company's own domain, then careers pages
    best_url = None
    best_score = -1

    # Try to guess the company's domain from the company name
    company_domain_hints: list[str] = []
    if company:
        # "Scotiabank" -> "scotiabank", "FDM Group" -> "fdmgroup" / "fdm"
        clean = company.lower().replace(" ", "").replace(".", "").replace(",", "")
        company_domain_hints.append(clean)
        # Also try just the first word
        first_word = company.lower().split()[0] if company else ""
        if first_word and len(first_word) > 2:
            company_domain_hints.append(first_word)

    for r in results:
        url = r["url"]
        score = 0

        # Skip LinkedIn/Indeed/Glassdoor aggregator results
        skip_domains = ["linkedin.com", "glassdoor.com", "indeed.com/career",
                        "indeed.com/q-", "youtube.com", "wikipedia.org",
                        "devpost.com", "prosple.com"]
        if any(d in url.lower() for d in skip_domains):
            continue

        url_lower = url.lower()
        result_title_lower = r["title"].lower()

        # ATS domain = very high confidence
        if _is_ats_url(url):
            score += 15

        # Company's own domain (e.g., scotiabank.com for Scotiabank)
        for hint in company_domain_hints:
            if hint in url_lower.split("/")[2] if len(url_lower.split("/")) > 2 else "":
                score += 12
                break

        # Has careers/jobs path
        if _is_careers_url(url):
            score += 5

        # Company name appears in URL or result title
        if company:
            company_lower = company.lower()
            if company_lower in url_lower:
                score += 4
            if company_lower in result_title_lower:
                score += 2

        # Title words appear in the URL or result title
        title_words = set(title.lower().split())
        matching_words = sum(1 for w in title_words
                           if len(w) > 3 and (w in url_lower or w in result_title_lower))
        score += matching_words

        # Bonus for specific job-posting URLs (not just generic careers page)
        if re.search(r"/job/|/jobs/\d|/positions?/\d|/posting/|/requisition/", url_lower):
            score += 3

        # Penalty for aggregator/third-party sites
        aggregators = ["jobright.ai", "bebee.com", "nodeflair.com",
                       "careercenter.", "ziprecruiter.com", "monster.com"]
        if any(a in url_lower for a in aggregators):
            score -= 3

        if score > best_score:
            best_score = score
            best_url = url

    # Only return if we have reasonable confidence
    if best_url and best_score >= 5:
        return best_url

    return None


# ---------------------------------------------------------------------------
# Tier 3: LLM extracts company → targeted ATS search
# ---------------------------------------------------------------------------

def _resolve_via_llm_search(title: str, full_description: str) -> str | None:
    """Use LLM to extract company name, then do a targeted search."""
    company = _extract_company_with_llm(full_description, title)
    if not company:
        log.info("Tier 3: LLM could not extract company name")
        return None

    log.info("Tier 3: LLM extracted company = '%s'", company)

    # Targeted ATS search for this company
    ats_queries = [
        f'"{company}" careers jobs "{title}"',
        f'site:greenhouse.io OR site:lever.co OR site:myworkdayjobs.com "{company}"',
        f'"{company}" job application "{title}"',
    ]

    for q in ats_queries:
        results = _search_duckduckgo(q, max_results=5)
        for r in results:
            url = r["url"]
            if "linkedin.com" in url:
                continue
            if _is_ats_url(url) or _is_careers_url(url):
                return url

    # Last resort: return the company's careers page if found
    results = _search_duckduckgo(f'"{company}" careers page', max_results=5)
    for r in results:
        url = r["url"]
        if "linkedin.com" not in url and _is_careers_url(url):
            return url

    return None


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

def resolve_single(url: str, title: str, full_description: str | None,
                   skip_tier1: bool = True) -> dict:
    """Resolve the application URL for a single job.

    Returns {"application_url": str | None, "tier": int | None, "company": str | None}
    """
    result = {"application_url": None, "tier": None, "company": None}
    t0 = time.time()

    # Extract company name — try multiple methods
    company = None

    # If it's a LinkedIn URL, scrape company from the page (cheap HTTP call)
    if "linkedin.com" in url:
        company = _scrape_linkedin_company(url)
        if company:
            log.info("Company from LinkedIn page: '%s'", company)

    # Fallback: regex extraction from JD
    if not company and full_description:
        company = _extract_company_from_jd(full_description, title)
        if company:
            log.info("Company from JD regex: '%s'", company)

    result["company"] = company

    # Tier 1: Scrape the original listing page for external apply redirect
    # Skipped by default — Playwright is slow, never succeeds on LinkedIn,
    # and conflicts with the apply stage which also uses Playwright/Chrome.
    if not skip_tier1:
        log.info("Tier 1: Scraping listing page %s", url[:60])
        apply_url = _resolve_from_listing_page(url)
        if apply_url:
            result["application_url"] = apply_url
            result["tier"] = 1
            log.info("Tier 1 resolved: %s (%.1fs)", apply_url[:80], time.time() - t0)
            return result

    # Tier 2: Web search with company + title
    log.info("Tier 2: Web search for '%s' at '%s'", title[:40], company or "unknown")
    apply_url = _resolve_via_search(title, company, full_description)
    if apply_url:
        result["application_url"] = apply_url
        result["tier"] = 2
        log.info("Tier 2 resolved: %s (%.1fs)", apply_url[:80], time.time() - t0)
        return result

    # Tier 3: LLM company extraction (if we still don't have company) → targeted search
    if full_description and not company:
        log.info("Tier 3: LLM-assisted company extraction")
        apply_url = _resolve_via_llm_search(title, full_description)
        if apply_url:
            result["application_url"] = apply_url
            result["tier"] = 3
            log.info("Tier 3 resolved: %s (%.1fs)", apply_url[:80], time.time() - t0)
            return result

    # Tier 3b: Even if we have a company name, try LLM for more targeted queries
    if full_description and company:
        log.info("Tier 3b: Targeted ATS search with known company '%s'", company)
        apply_url = _resolve_via_llm_search(title, full_description)
        if apply_url:
            result["application_url"] = apply_url
            result["tier"] = 3
            log.info("Tier 3b resolved: %s (%.1fs)", apply_url[:80], time.time() - t0)
            return result

    log.warning("All tiers failed for: %s (%.1fs)", title[:50], time.time() - t0)
    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_url_resolution(limit: int = 50) -> dict:
    """Resolve missing application URLs in batch.

    Targets qualified jobs (tailored_resume_path set) that have no application_url.
    Processes in batches, committing after each job so progress is saved.

    Args:
        limit: Max jobs to process per batch.

    Returns:
        Dict with stats: processed, resolved, failed, tiers breakdown.
    """
    conn = init_db()
    now = datetime.now(timezone.utc).isoformat()

    # Get jobs missing application_url that have tailored resumes
    rows = conn.execute(
        "SELECT url, title, full_description FROM jobs "
        "WHERE tailored_resume_path IS NOT NULL "
        "AND (application_url IS NULL OR application_url = '') "
        "ORDER BY fit_score DESC "
        "LIMIT ?",
        (limit,),
    ).fetchall()

    stats = {
        "processed": 0,
        "resolved": 0,
        "failed": 0,
        "tiers": {1: 0, 2: 0, 3: 0},
    }

    if not rows:
        log.info("No jobs need URL resolution.")
        return stats

    log.info("URL resolver: %d jobs to process", len(rows))

    for i, row in enumerate(rows):
        job_url, title, full_desc = row[0], row[1], row[2]
        stats["processed"] += 1

        log.info("[%d/%d] Resolving: %s", i + 1, len(rows), title[:50] if title else job_url[:50])

        try:
            result = resolve_single(job_url, title or "", full_desc)
            apply_url = result.get("application_url")

            if apply_url:
                conn.execute(
                    "UPDATE jobs SET application_url = ? WHERE url = ?",
                    (apply_url, job_url),
                )
                conn.commit()
                stats["resolved"] += 1
                tier = result.get("tier", 0)
                if tier in stats["tiers"]:
                    stats["tiers"][tier] += 1
                log.info("  RESOLVED (T%d): %s", tier, apply_url[:80])
            else:
                stats["failed"] += 1
                log.info("  FAILED: no URL found")

        except Exception as e:
            stats["failed"] += 1
            log.error("  ERROR: %s", e)

        # Rate limit between jobs to avoid search bans
        if i < len(rows) - 1:
            time.sleep(2.0)

    log.info("URL resolution complete: %d processed, %d resolved, %d failed | T1=%d T2=%d T3=%d",
             stats["processed"], stats["resolved"], stats["failed"],
             stats["tiers"][1], stats["tiers"][2], stats["tiers"][3])

    return stats
