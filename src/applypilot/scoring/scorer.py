"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from applypilot.config import RESUME_PATH, load_profile
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.llm import get_client

log = logging.getLogger(__name__)


# ── Scoring Prompt ────────────────────────────────────────────────────────

SCORE_PROMPT = """You are an ATS-aware job fit evaluator. Given a candidate's resume and a job description, perform a thorough ATS-style analysis and score the match.

## ANALYSIS STEPS (do all internally before scoring):
1. Identify the top 10 keywords/phrases from the JD that an ATS would scan for
2. Check which of those keywords appear in the resume (exact or close match)
3. Identify skills gaps that would hurt the application
4. Check industry-specific terminology alignment
5. Evaluate seniority-level language match (junior/senior/lead)
6. Assess technical requirements coverage
7. Note soft skills from the JD that are present or missing

## SCORING CRITERIA:
- 9-10: ATS would rank in top 5%. 8+ of 10 critical keywords present. Direct experience in nearly all required skills. Seniority match.
- 7-8: ATS would rank in top 20%. 6-7 of 10 keywords present. Most required skills covered, minor gaps easily bridged by tailoring.
- 5-6: ATS would flag as borderline. 4-5 keywords present. Some relevant skills but missing key requirements. Tailoring could help significantly.
- 3-4: ATS would likely reject. Under 4 keywords. Significant skill gaps, different domain or seniority level.
- 1-2: ATS auto-reject. Completely different field, experience level, or tech stack.

## IMPORTANT FACTORS:
- Weight technical skills and exact keyword matches heavily (this is what ATS systems do)
- Consider transferable experience (automation, scripting, API work) as bridgeable gaps
- Factor in the candidate's project experience as evidence of skills
- Be realistic about experience level vs job requirements
- Consider if the resume could be tailored to score 90+ on an ATS for this JD

RESPOND IN EXACTLY THIS FORMAT (no other text):
SCORE: [1-10]
KEYWORDS: [comma-separated top ATS keywords from the JD that match or could match the candidate]
REASONING: [2-3 sentences: match percentage, key gaps, and whether tailoring could bridge them]"""


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    score = 0
    keywords = ""
    reasoning = response

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = int(re.search(r"\d+", line).group())
                score = max(1, min(10, score))
            except (AttributeError, ValueError):
                score = 0
        elif line.startswith("KEYWORDS:"):
            keywords = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return {"score": score, "keywords": keywords, "reasoning": reasoning}


def score_job(resume_text: str, job: dict) -> dict:
    """Score a single job against the resume.

    Args:
        resume_text: The candidate's full resume text.
        job: Job dict with keys: title, site, location, full_description.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    messages = [
        {"role": "system", "content": SCORE_PROMPT},
        {"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"},
    ]

    try:
        client = get_client()
        response = client.chat(messages, max_tokens=2048, temperature=0.0)
        return _parse_score_response(response)
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {"score": 0, "keywords": "", "reasoning": f"LLM error: {e}"}


def run_scoring(limit: int = 0, rescore: bool = False, workers: int = 3) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).
        workers: Number of parallel scoring threads (default 3).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list}
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", limit=limit)

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    log.info("Scoring %d jobs (%d workers)...", len(jobs), workers)
    t0 = time.time()
    total_scored = 0
    total_errors = 0
    _pending: list[dict] = []
    _lock = threading.Lock()
    COMMIT_EVERY = 10

    def _flush(pending: list[dict]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        for r in pending:
            conn.execute(
                "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ? WHERE url = ?",
                (r["score"], f"{r['keywords']}\n{r['reasoning']}", now, r["url"]),
            )
        conn.commit()

    def _on_result(result: dict, idx: int) -> None:
        nonlocal total_scored, total_errors
        with _lock:
            total_scored += 1
            if result["score"] == 0:
                total_errors += 1
            _pending.append(result)
            log.info("[%d/%d] score=%d  %s", idx, len(jobs), result["score"], result.get("title", "?")[:60])
            if len(_pending) >= COMMIT_EVERY:
                _flush(_pending.copy())
                _pending.clear()

    def _score_one(job: dict, idx: int) -> tuple[dict, int]:
        result = score_job(resume_text, job)
        result["url"] = job["url"]
        result["title"] = job.get("title", "?")
        return result, idx

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_score_one, job, i + 1): job for i, job in enumerate(jobs)}
            for future in as_completed(futures):
                result, idx = future.result()
                _on_result(result, idx)
    else:
        for i, job in enumerate(jobs):
            result, idx = _score_one(job, i + 1)
            _on_result(result, idx)

    # Flush remaining
    with _lock:
        if _pending:
            _flush(_pending)

    elapsed = time.time() - t0
    log.info("Done: %d scored in %.1fs (%.1f jobs/sec)", total_scored, elapsed, total_scored / elapsed if elapsed > 0 else 0)

    # Score distribution
    dist = conn.execute("""
        SELECT fit_score, COUNT(*) FROM jobs
        WHERE fit_score IS NOT NULL
        GROUP BY fit_score ORDER BY fit_score DESC
    """).fetchall()
    distribution = [(row[0], row[1]) for row in dist]

    return {
        "scored": total_scored,
        "errors": total_errors,
        "elapsed": elapsed,
        "distribution": distribution,
    }
