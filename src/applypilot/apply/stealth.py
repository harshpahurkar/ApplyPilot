"""Anti-rate-limit evasion layer for Copilot CLI interactions.

Provides human-like pacing, automation signal removal, model rotation
across free-tier (0x multiplier) models, proxy support, and per-window
request budgeting.  The goal is to make automated Copilot CLI calls
look indistinguishable from a human developer chatting interactively.

Key insights from GitHub rate-limit mechanics (March 2026):
- Rate limits are token-throughput based per sliding time window.
- ``CI=1``, ``TERM=dumb``, ``NO_COLOR=1`` are telemetry signals that
  GitHub can use to apply stricter limits to automated/CI usage.
- ``--output-format json`` signals programmatic consumption.
- GPT-4.1 / GPT-4o / GPT-5 mini carry a 0x premium multiplier (free
  on all paid plans).  Claude models carry 1-3x multipliers.
- "Auto" model selection is exempt from per-model rate limits.
- Short-lived sessions (one CLI call per task) vs. long-running
  conversational threads is a strong behavioral signal.
"""

from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Human-like pacing
# ---------------------------------------------------------------------------

class HumanPacer:
    """Generate natural-feeling delays that mimic human interaction cadence.

    Delays follow a truncated Gaussian distribution (never negative,
    capped at 2× the mean) to avoid both machine-perfect intervals and
    extreme outliers.

    Settable via environment:
        APPLYPILOT_STEALTH_JOB_DELAY_MIN   (default 45)
        APPLYPILOT_STEALTH_JOB_DELAY_MAX   (default 120)
        APPLYPILOT_STEALTH_LLM_DELAY_MIN   (default 4)
        APPLYPILOT_STEALTH_LLM_DELAY_MAX   (default 12)
        APPLYPILOT_STEALTH_WARMUP_EXTRA    (default 15)
        APPLYPILOT_STEALTH_ENABLED         (default 1)
    """

    def __init__(self) -> None:
        self.enabled = _truthy(os.environ.get("APPLYPILOT_STEALTH_ENABLED", "1"))
        self._job_min = _env_float("APPLYPILOT_STEALTH_JOB_DELAY_MIN", 45.0)
        self._job_max = _env_float("APPLYPILOT_STEALTH_JOB_DELAY_MAX", 120.0)
        self._llm_min = _env_float("APPLYPILOT_STEALTH_LLM_DELAY_MIN", 4.0)
        self._llm_max = _env_float("APPLYPILOT_STEALTH_LLM_DELAY_MAX", 12.0)
        self._warmup = _env_float("APPLYPILOT_STEALTH_WARMUP_EXTRA", 15.0)
        self._jobs_completed = 0
        self._lock = threading.Lock()
        # Adaptive backoff: after timeouts, increase delays to let the
        # provider recover.  Decays back to normal on success.
        self._timeout_penalty: float = 0.0
        self._max_penalty = _env_float("APPLYPILOT_STEALTH_MAX_PENALTY", 60.0)

    def _gaussian_delay(self, low: float, high: float) -> float:
        """Truncated Gaussian between *low* and *high*."""
        mean = (low + high) / 2
        sigma = (high - low) / 4  # 95% falls within range
        delay = random.gauss(mean, sigma)
        return max(low, min(high, delay))

    def inter_job_delay(self) -> float:
        """Seconds to wait between apply jobs (mimics tab-switching, reading)."""
        if not self.enabled:
            return 0.0
        with self._lock:
            base = self._gaussian_delay(self._job_min, self._job_max)
            # First few jobs get extra warmup to look like a slow start
            if self._jobs_completed < 3:
                base += self._warmup * (3 - self._jobs_completed) / 3
            self._jobs_completed += 1
            return base

    def inter_llm_delay(self) -> float:
        """Seconds to wait between Copilot LLM calls (scoring/tailoring)."""
        if not self.enabled:
            return 0.0
        with self._lock:
            base = self._gaussian_delay(self._llm_min, self._llm_max)
            return base + self._timeout_penalty

    def report_timeout(self) -> None:
        """Called after a Copilot timeout — increase delays adaptively."""
        with self._lock:
            # Each timeout doubles the penalty (starting from 10s), capped
            self._timeout_penalty = min(
                max(self._timeout_penalty * 2, 10.0),
                self._max_penalty,
            )
            log.info(
                "[stealth] timeout penalty raised to %.0fs",
                self._timeout_penalty,
            )

    def report_success(self) -> None:
        """Called after a successful LLM call — decay the penalty."""
        with self._lock:
            if self._timeout_penalty > 0:
                self._timeout_penalty = max(self._timeout_penalty * 0.5, 0.0)
                if self._timeout_penalty < 1.0:
                    self._timeout_penalty = 0.0
                log.debug(
                    "[stealth] timeout penalty decayed to %.0fs",
                    self._timeout_penalty,
                )

    def reading_delay(self, response_length: int = 500) -> float:
        """Simulate time spent reading a response (proportional to length)."""
        if not self.enabled:
            return 0.0
        # ~200 words per minute ≈ 1000 chars/minute → 0.06s per char
        # But cap between 1-8 seconds
        base = min(max(response_length * 0.003, 1.0), 8.0)
        return base + random.uniform(0, 2.0)

    def sleep_inter_job(self) -> None:
        """Block for a human-like inter-job interval."""
        delay = self.inter_job_delay()
        if delay > 0:
            log.info("[stealth] inter-job pause %.1fs", delay)
            time.sleep(delay)

    def sleep_inter_llm(self) -> None:
        """Block for a human-like inter-LLM-call interval."""
        delay = self.inter_llm_delay()
        if delay > 0:
            log.debug("[stealth] inter-LLM pause %.1fs", delay)
            time.sleep(delay)


# Module-level singleton
_pacer: HumanPacer | None = None
_pacer_lock = threading.Lock()


def get_pacer() -> HumanPacer:
    """Return the global pacer singleton."""
    global _pacer
    if _pacer is None:
        with _pacer_lock:
            if _pacer is None:
                _pacer = HumanPacer()
    return _pacer


# ---------------------------------------------------------------------------
# Automation signal removal
# ---------------------------------------------------------------------------

# These environment variables are telemetry signals that mark a process as
# automated / CI.  GitHub can (and likely does) use them to apply stricter
# rate limits or flag sessions for abuse detection.
_AUTOMATION_ENV_KEYS = {"CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS"}

# These env vars are also visible but less suspicious; we keep them to avoid
# breaking terminal rendering in non-interactive mode.
_SAFE_TERMINAL_VARS = {"TERM", "NO_COLOR", "FORCE_COLOR"}


def clean_env_for_copilot(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of the environment with automation signals removed.

    Removes CI=1 and similar flags that tell Copilot's backend this is an
    automated session.  Keeps safe terminal rendering vars that don't
    affect rate-limit classification.
    """
    out = dict(env) if env is not None else dict(os.environ)

    # Remove automation markers
    for key in _AUTOMATION_ENV_KEYS:
        out.pop(key, None)

    # Set terminal vars to look like a normal interactive session
    out.pop("TERM", None)  # Let it inherit the real terminal type
    out.pop("NO_COLOR", None)  # Interactive sessions have color
    # Don't set TERM=dumb — that's a dead giveaway

    return out


# ---------------------------------------------------------------------------
# Free-tier model rotation (0x premium multiplier)
# ---------------------------------------------------------------------------

# Models that carry a 0× premium multiplier on GitHub Copilot
# (as of March 2026).  Using these does NOT consume premium request quota.
FREE_TIER_MODELS: list[str] = [
    "gpt-4.1",
    "gpt-4o",
    "gpt-5-mini",
    "gpt-4.1-mini",
]

# Premium models and their multipliers (for budget awareness)
PREMIUM_MODEL_MULTIPLIERS: dict[str, float] = {
    "claude-sonnet-4": 1.0,
    "claude-sonnet-4.6": 1.0,
    "claude-opus-4": 3.0,
    "claude-opus-4.6": 3.0,
    "o3": 1.0,
    "o4-mini": 0.33,
    "gemini-2.5-pro": 1.0,
}


class ModelRotator:
    """Rotate between free-tier models to distribute load and avoid
    per-model rate limits.

    The "Auto" model selection on Copilot is exempt from per-model limits
    but we can't specify "auto" via CLI ``--model``.  Instead, we rotate
    through the 0x-multiplier models.

    Environment:
        APPLYPILOT_STEALTH_MODEL_POOL  — comma-separated override
        APPLYPILOT_STEALTH_ROTATE      — 1/0 to enable/disable (default 1)
    """

    def __init__(self) -> None:
        self.enabled = _truthy(os.environ.get("APPLYPILOT_STEALTH_ROTATE", "1"))
        pool_raw = os.environ.get("APPLYPILOT_STEALTH_MODEL_POOL", "").strip()
        if pool_raw:
            self.pool = [m.strip() for m in pool_raw.split(",") if m.strip()]
        else:
            self.pool = list(FREE_TIER_MODELS)
        self._index = 0
        self._lock = threading.Lock()

    def next_model(self) -> str:
        """Return the next model in the rotation pool."""
        if not self.enabled or not self.pool:
            return FREE_TIER_MODELS[0]
        with self._lock:
            model = self.pool[self._index % len(self.pool)]
            self._index += 1
            return model

    def is_free_tier(self, model: str) -> bool:
        """Check if a model name is in the free tier."""
        return model.lower() in {m.lower() for m in FREE_TIER_MODELS}

    def premium_multiplier(self, model: str) -> float:
        """Return the premium multiplier for a model (0.0 for free tier)."""
        if self.is_free_tier(model):
            return 0.0
        return PREMIUM_MODEL_MULTIPLIERS.get(model.lower(), 1.0)


_rotator: ModelRotator | None = None
_rotator_lock = threading.Lock()


def get_rotator() -> ModelRotator:
    """Return the global model rotator singleton."""
    global _rotator
    if _rotator is None:
        with _rotator_lock:
            if _rotator is None:
                _rotator = ModelRotator()
    return _rotator


# ---------------------------------------------------------------------------
# Request budget tracker (per sliding window)
# ---------------------------------------------------------------------------

@dataclass
class RequestBudget:
    """Track request volume per sliding time window.

    When the budget is near exhaustion, callers should increase delays.

    Environment:
        APPLYPILOT_STEALTH_BUDGET_WINDOW    — window size in seconds (default 3600)
        APPLYPILOT_STEALTH_BUDGET_MAX       — max requests per window (default 40)
        APPLYPILOT_STEALTH_BUDGET_SOFT      — soft limit (start slowing) (default 25)
    """
    window_seconds: float = field(default_factory=lambda: _env_float("APPLYPILOT_STEALTH_BUDGET_WINDOW", 3600))
    max_requests: int = field(default_factory=lambda: _env_int("APPLYPILOT_STEALTH_BUDGET_MAX", 40))
    soft_limit: int = field(default_factory=lambda: _env_int("APPLYPILOT_STEALTH_BUDGET_SOFT", 25))
    _timestamps: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self) -> None:
        """Record a request."""
        now = time.time()
        with self._lock:
            self._timestamps.append(now)
            self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def count_in_window(self) -> int:
        """Number of requests in the current window."""
        now = time.time()
        with self._lock:
            self._prune(now)
            return len(self._timestamps)

    def utilization(self) -> float:
        """Current utilization as a fraction (0.0 to 1.0+)."""
        return self.count_in_window() / max(self.max_requests, 1)

    def should_slow_down(self) -> bool:
        """True if we're past the soft limit and should increase delays."""
        return self.count_in_window() >= self.soft_limit

    def is_exhausted(self) -> bool:
        """True if we've hit the max budget for this window."""
        return self.count_in_window() >= self.max_requests

    def adaptive_delay(self) -> float:
        """Return an additional delay (seconds) based on current utilization.

        Returns 0 when below soft limit, scales up to 60s at max budget.
        """
        count = self.count_in_window()
        if count < self.soft_limit:
            return 0.0
        # Linear scale from 0 at soft_limit to 60 at max_requests
        over = count - self.soft_limit
        range_size = max(self.max_requests - self.soft_limit, 1)
        fraction = min(over / range_size, 1.0)
        return fraction * 60.0

    def wait_if_exhausted(self) -> float:
        """If budget is exhausted, block until the oldest request expires.

        Returns seconds waited (0 if not exhausted).
        """
        with self._lock:
            now = time.time()
            self._prune(now)
            if len(self._timestamps) < self.max_requests:
                return 0.0
            # Wait for the oldest request to fall out of the window
            oldest = self._timestamps[0]
            wait = (oldest + self.window_seconds) - now + 1.0
        if wait > 0:
            log.warning("[stealth] budget exhausted, waiting %.0fs", wait)
            time.sleep(wait)
            return wait
        return 0.0


_budget: RequestBudget | None = None
_budget_lock = threading.Lock()


def get_budget() -> RequestBudget:
    """Return the global request budget singleton."""
    global _budget
    if _budget is None:
        with _budget_lock:
            if _budget is None:
                _budget = RequestBudget()
    return _budget


# ---------------------------------------------------------------------------
# Proxy support
# ---------------------------------------------------------------------------

def apply_proxy_env(env: dict[str, str]) -> dict[str, str]:
    """Inject proxy settings into subprocess environment if configured.

    Environment:
        APPLYPILOT_PROXY_URL — HTTP(S) proxy URL (e.g. http://proxy:8080)
        APPLYPILOT_PROXY_AUTH — Optional Basic auth (user:pass)

    The proxy is injected as standard HTTP_PROXY / HTTPS_PROXY vars
    which Node.js (and thus Copilot CLI) will honor.
    """
    proxy_url = os.environ.get("APPLYPILOT_PROXY_URL", "").strip()
    if not proxy_url:
        return env

    proxy_auth = os.environ.get("APPLYPILOT_PROXY_AUTH", "").strip()
    if proxy_auth and "@" not in proxy_url:
        # Insert auth into the URL: http://user:pass@host:port
        scheme_end = proxy_url.find("://")
        if scheme_end >= 0:
            proxy_url = (
                proxy_url[: scheme_end + 3]
                + proxy_auth + "@"
                + proxy_url[scheme_end + 3:]
            )

    env["HTTP_PROXY"] = proxy_url
    env["HTTPS_PROXY"] = proxy_url
    env["http_proxy"] = proxy_url
    env["https_proxy"] = proxy_url
    log.debug("[stealth] proxy configured: %s", proxy_url[:30] + "...")
    return env


# ---------------------------------------------------------------------------
# Composite: prepare environment for a Copilot CLI call
# ---------------------------------------------------------------------------

def prepare_copilot_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Full environment preparation: remove automation signals + apply proxy.

    Call this instead of manually setting CI/TERM/NO_COLOR.
    """
    out = clean_env_for_copilot(env)
    out = apply_proxy_env(out)
    return out
