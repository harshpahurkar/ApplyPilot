"""Rezzy API client: async resume generation via rezzy.dev.

Primary resume tailoring engine. Produces its own PDF — bypasses the
local LLM prompt/coercion/LaTeX pipeline entirely.

Fallback: when credits are exhausted (403), returns None so the caller
can fall back to the Copilot-based tailor pipeline.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.rezzy.dev/v1"
POLL_INTERVAL = 5  # seconds between status polls
POLL_TIMEOUT = 180  # max seconds to wait for completion


class RezzyCreditsExhausted(Exception):
    """Raised when the Rezzy API returns 403 (insufficient credits)."""


class RezzyError(Exception):
    """Generic Rezzy API error."""


def _get_api_key() -> str:
    key = os.environ.get("REZZY_API_KEY", "").strip()
    if not key:
        raise RezzyError("REZZY_API_KEY not set in environment")
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


def _handle_error(resp: requests.Response) -> None:
    """Raise appropriate exception for error responses."""
    if resp.status_code == 403:
        raise RezzyCreditsExhausted("Insufficient Rezzy credits")
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "60"))
        log.warning("Rezzy rate limited — sleeping %ds", retry_after)
        time.sleep(retry_after)
        return  # caller should retry
    if resp.status_code >= 400:
        raise RezzyError(f"Rezzy API error {resp.status_code}: {resp.text}")


def create_resume(title: str, job_description: str, company_url: str | None = None,
                   job_url: str | None = None) -> str:
    """Queue a resume generation job on Rezzy.

    Args:
        title: Job title (used as resume title).
        job_description: Full job description text.
        company_url: Optional company website URL for deeper research.
        job_url: Optional job listing URL (used to make title unique on 409).

    Returns:
        Resume ID (UUID) for polling.

    Raises:
        RezzyCreditsExhausted: If credits are depleted.
        RezzyError: On other API errors.
    """
    # Make title unique per job to avoid 409 conflicts
    suffix = hashlib.sha1((job_url or title).encode()).hexdigest()[:6]
    unique_title = f"{title[:245]} ({suffix})"

    payload: dict = {
        "title": unique_title,
        "job_description": job_description,
    }
    if company_url:
        # Validate it looks like a real URL
        parsed = urlparse(company_url)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            payload["company_url"] = company_url

    for attempt in range(3):
        resp = requests.post(
            f"{BASE_URL}/resume/create",
            headers=_headers(),
            json=payload,
            timeout=30,
        )
        if resp.status_code == 429:
            _handle_error(resp)
            continue
        # If company_url caused a 400, retry without it
        if resp.status_code == 400 and "company_url" in payload:
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            msg = body.get("message", "")
            if "company" in msg.lower() or "url" in msg.lower():
                log.info("Rezzy rejected company_url — retrying without it")
                del payload["company_url"]
                continue
        break
    else:
        raise RezzyError("Rezzy rate limited after 3 retries")

    _handle_error(resp)

    data = resp.json().get("data", {})
    resume_id = data.get("id")
    if not resume_id:
        raise RezzyError(f"No resume ID in response: {resp.text}")

    log.info("Rezzy resume queued: id=%s title=%s", resume_id, payload["title"][:50])
    return resume_id


def poll_resume(resume_id: str) -> dict:
    """Poll until the resume is ready or failed.

    Returns:
        Rezzy data dict with keys: id, resume_title, status, pdf_url, etc.

    Raises:
        RezzyCreditsExhausted: If credits check fails during poll.
        RezzyError: On timeout or API error.
    """
    start = time.time()
    last_stage = ""

    while time.time() - start < POLL_TIMEOUT:
        resp = requests.get(
            f"{BASE_URL}/resume/{resume_id}",
            headers=_headers(),
            timeout=30,
        )
        if resp.status_code == 429:
            _handle_error(resp)
            continue
        _handle_error(resp)

        data = resp.json().get("data", {})
        status = data.get("status", "").upper()
        stage = data.get("stage", "")

        if stage != last_stage:
            log.info("Rezzy %s: %s — %s", resume_id[:8], status, stage)
            last_stage = stage

        if status == "SUCCESS":
            return data
        if status == "FAILED":
            raise RezzyError(f"Rezzy resume generation failed: {data}")

        time.sleep(POLL_INTERVAL)

    raise RezzyError(f"Rezzy poll timeout after {POLL_TIMEOUT}s for {resume_id}")


def download_pdf(pdf_url: str, dest_path: Path) -> Path:
    """Download the generated PDF to a local path.

    Args:
        pdf_url: URL from Rezzy's pdf_url field.
        dest_path: Local path to save the PDF.

    Returns:
        The dest_path on success.
    """
    resp = requests.get(pdf_url, timeout=60)
    resp.raise_for_status()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(resp.content)
    log.info("Rezzy PDF saved: %s (%d bytes)", dest_path.name, len(resp.content))
    return dest_path


def rezzy_tailor_resume(
    job: dict,
    output_dir: Path,
    file_prefix: str,
) -> dict | None:
    """End-to-end: create resume on Rezzy, poll, download PDF.

    Args:
        job: Job dict with title, site, full_description, url, application_url.
        output_dir: Directory to save the PDF (usually TAILORED_DIR).
        file_prefix: Filename prefix (no extension).

    Returns:
        Dict with:
            "pdf_path": str — path to the downloaded PDF
            "tex_path": str — path to a stub .tex file (for DB compatibility)
            "source": "rezzy"
        Or None if credits are exhausted (caller should fall back).

    Raises:
        RezzyError: On non-credit API errors that are not recoverable.
    """
    title = job.get("title", "Software Engineer")
    jd = job.get("full_description", "")
    if not jd:
        log.warning("No job description for Rezzy — skipping to fallback")
        return None

    # Try to extract company URL from application_url or url
    company_url = None
    for url_field in ("application_url", "url"):
        raw_url = job.get(url_field, "")
        if raw_url:
            parsed = urlparse(raw_url)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                # Use the base domain as company URL
                company_url = f"{parsed.scheme}://{parsed.netloc}"
                break

    try:
        resume_id = create_resume(title, jd, company_url, job_url=job.get("url"))
        data = poll_resume(resume_id)
    except RezzyCreditsExhausted:
        log.warning("Rezzy credits exhausted — falling back to Copilot tailor")
        return None

    pdf_url = data.get("pdf_url")
    if not pdf_url:
        log.error("Rezzy resume completed but no pdf_url: %s", data)
        raise RezzyError("Rezzy resume completed but no PDF URL returned")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{file_prefix}.pdf"
    download_pdf(pdf_url, pdf_path)

    # Create a stub .tex file so the DB path convention works
    # (apply worker finds PDF by changing .tex extension to .pdf)
    tex_path = output_dir / f"{file_prefix}.tex"
    tex_path.write_text(
        f"% Rezzy-generated resume — PDF downloaded from API\n"
        f"% Resume ID: {data.get('id', 'unknown')}\n"
        f"% Title: {title}\n"
        f"% Source: rezzy.dev\n",
        encoding="utf-8",
    )

    # Also save the job description for reference
    job_path = output_dir / f"{file_prefix}_JOB.txt"
    job_desc = (
        f"Title: {title}\n"
        f"Company: {job.get('site', 'N/A')}\n"
        f"Location: {job.get('location', 'N/A')}\n"
        f"Score: {job.get('fit_score', 'N/A')}\n"
        f"URL: {job.get('url', 'N/A')}\n"
        f"Source: Rezzy API\n\n"
        f"{jd}"
    )
    job_path.write_text(job_desc, encoding="utf-8")

    return {
        "pdf_path": str(pdf_path),
        "tex_path": str(tex_path),
        "source": "rezzy",
    }


def is_rezzy_enabled() -> bool:
    """Check if Rezzy is configured and enabled."""
    enabled = os.environ.get("APPLYPILOT_USE_REZZY", "").strip().lower() in {"1", "true", "yes", "on"}
    has_key = bool(os.environ.get("REZZY_API_KEY", "").strip())
    return enabled and has_key
