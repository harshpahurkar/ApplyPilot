"""Structured error classification for ApplyPilot.

Provides a unified ErrorCategory enum for classifying LLM and apply errors
into actionable categories (transient vs permanent), replacing ad-hoc
string matching across modules.
"""

from enum import Enum


class ErrorCategory(Enum):
    """Unified error classification across all ApplyPilot subsystems."""

    # Transient — safe to retry
    TRANSIENT_RATE_LIMIT = "transient_rate_limit"
    TRANSIENT_NETWORK = "transient_network"
    TRANSIENT_TIMEOUT = "transient_timeout"
    TRANSIENT_OVERLOADED = "transient_overloaded"

    # Permanent — do not retry, skip to next provider or fail
    PERMANENT_AUTH = "permanent_auth"
    PERMANENT_QUOTA = "permanent_quota"
    PERMANENT_INPUT = "permanent_input"
    PERMANENT_NOT_FOUND = "permanent_not_found"

    # Apply-specific permanent errors
    PERMANENT_CAPTCHA = "permanent_captcha"
    PERMANENT_SSO = "permanent_sso"
    PERMANENT_EXPIRED = "permanent_expired"
    PERMANENT_ALREADY_APPLIED = "permanent_already_applied"
    PERMANENT_NOT_ELIGIBLE = "permanent_not_eligible"
    PERMANENT_SITE_BLOCKED = "permanent_site_blocked"

    # Apply-specific transient errors
    TRANSIENT_LOGIN = "transient_login"
    TRANSIENT_PAGE_ERROR = "transient_page_error"

    UNKNOWN = "unknown"

    @property
    def is_transient(self) -> bool:
        return self.value.startswith("transient_")

    @property
    def is_permanent(self) -> bool:
        return self.value.startswith("permanent_")


def classify_llm_error(
    status_code: int | None = None,
    error_text: str = "",
) -> ErrorCategory:
    """Classify an LLM API error into a structured category.

    Args:
        status_code: HTTP status code (if available).
        error_text: Error message text for deeper inspection.

    Returns:
        ErrorCategory enum value.
    """
    text = (error_text or "").lower()

    # Rate limiting
    if status_code == 429 or "rate limit" in text or "(rate_limit)" in text:
        return ErrorCategory.TRANSIENT_RATE_LIMIT

    # Overloaded
    if status_code == 529 or status_code == 503 or "overloaded" in text:
        return ErrorCategory.TRANSIENT_OVERLOADED

    # Auth failures
    if status_code == 401 or "unauthorized" in text or "invalid api key" in text:
        return ErrorCategory.PERMANENT_AUTH
    if status_code == 403 and "forbidden" in text:
        return ErrorCategory.PERMANENT_AUTH

    # Quota / payment
    if status_code == 402 or "no credits" in text or "quota" in text:
        return ErrorCategory.PERMANENT_QUOTA

    # Input errors (prompt too long, bad request)
    if status_code == 400 or "prompt is too long" in text:
        return ErrorCategory.PERMANENT_INPUT

    # Network errors
    if any(kw in text for kw in ("timeout", "timed out", "timeoutexception")):
        return ErrorCategory.TRANSIENT_TIMEOUT
    if any(kw in text for kw in (
        "connection", "econnreset", "epipe", "read error", "connect error",
        "remote protocol", "network",
    )):
        return ErrorCategory.TRANSIENT_NETWORK

    return ErrorCategory.UNKNOWN


def classify_apply_error(status: str, error_text: str = "") -> ErrorCategory:
    """Classify an apply pipeline error into a structured category.

    Args:
        status: The status string from agent output parsing.
        error_text: Additional error details.

    Returns:
        ErrorCategory enum value.
    """
    s = (status or "").lower().replace("failed:", "").strip()
    text = (error_text or "").lower()

    # Permanent apply errors (human-impossible)
    if s in ("captcha", "cloudflare_blocked"):
        return ErrorCategory.PERMANENT_CAPTCHA
    if s in ("sso_required", "mfa_required"):
        return ErrorCategory.PERMANENT_SSO
    if s in ("expired", "not_found", "position_closed"):
        return ErrorCategory.PERMANENT_EXPIRED
    if s == "already_applied":
        return ErrorCategory.PERMANENT_ALREADY_APPLIED
    if s in ("not_eligible_location", "not_eligible_salary", "not_a_job_application"):
        return ErrorCategory.PERMANENT_NOT_ELIGIBLE
    if s in ("site_blocked", "host_unreachable", "page_load_failed", "browser_connection_error"):
        return ErrorCategory.PERMANENT_SITE_BLOCKED

    # Transient apply errors (can retry)
    if s in ("login_issue", "email_verification"):
        return ErrorCategory.TRANSIENT_LOGIN
    if s in ("timeout", "page_error", "early_stage_stuck"):
        return ErrorCategory.TRANSIENT_PAGE_ERROR
    if s in ("no_result_line",):
        return ErrorCategory.TRANSIENT_PAGE_ERROR
    if s in ("rate_limited", "model_unavailable"):
        return ErrorCategory.TRANSIENT_RATE_LIMIT

    # Check error text for additional hints
    if "rate limit" in text or "rate_limit" in text:
        return ErrorCategory.TRANSIENT_RATE_LIMIT
    if "timeout" in text:
        return ErrorCategory.TRANSIENT_TIMEOUT
    if "captcha" in text:
        return ErrorCategory.PERMANENT_CAPTCHA
    if "expired" in text or "no longer" in text:
        return ErrorCategory.PERMANENT_EXPIRED

    return ErrorCategory.UNKNOWN
