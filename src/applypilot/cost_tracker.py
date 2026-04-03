"""Per-model LLM cost tracking for ApplyPilot sessions.

Tracks input/output tokens and estimated USD cost per provider/model,
persists across session restarts, and prints a formatted summary on exit.

Inspired by Claude Code's cost-tracker.ts architecture.
"""

import atexit
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from applypilot.config import APP_DIR

log = logging.getLogger(__name__)

# Approximate pricing per 1M tokens (input / output) — update as needed.
# These are ballpark figures for cost estimation; exact pricing varies.
_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1M_tokens, output_per_1M_tokens)
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.5-flash-preview-04-17": (0.15, 0.60),
    "deepseek-chat": (0.14, 0.28),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-sonnet-4.6": (3.00, 15.00),
    "claude-haiku-4.5": (0.80, 4.00),
}

COST_FILE = APP_DIR / "session_costs.json"


@dataclass
class ModelUsage:
    """Accumulated usage for a single model."""
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    errors: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
            "errors": self.errors,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class SessionCosts:
    """Full session cost state."""
    session_id: str = ""
    started_at: float = 0.0
    total_cost_usd: float = 0.0
    total_calls: int = 0
    total_errors: int = 0
    model_usage: dict[str, ModelUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_call(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: bool = False,
    ) -> float:
        """Record an LLM call and return the estimated cost in USD."""
        key = f"{provider}/{model}"
        cost = _estimate_cost(model, input_tokens, output_tokens)

        with self._lock:
            usage = self.model_usage.get(key)
            if usage is None:
                usage = ModelUsage()
                self.model_usage[key] = usage

            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens
            usage.calls += 1
            usage.cost_usd += cost
            if error:
                usage.errors += 1
                self.total_errors += 1

            self.total_calls += 1
            self.total_cost_usd += cost

        return cost

    def format_summary(self) -> str:
        """Format a human-readable cost summary."""
        lines = []
        duration = time.time() - self.started_at if self.started_at else 0

        lines.append(f"Session cost: ${self.total_cost_usd:.4f}")
        lines.append(f"Total calls: {self.total_calls} ({self.total_errors} errors)")
        if duration > 0:
            lines.append(f"Duration: {int(duration)}s")

        if self.model_usage:
            lines.append("")
            lines.append("Per-model breakdown:")
            for key, usage in sorted(self.model_usage.items()):
                lines.append(
                    f"  {key}: {usage.calls} calls, "
                    f"{usage.input_tokens:,} in / {usage.output_tokens:,} out, "
                    f"${usage.cost_usd:.4f}"
                )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "model_usage": {k: v.to_dict() for k, v in self.model_usage.items()},
        }

    def save(self) -> None:
        """Persist current session costs to disk."""
        try:
            COST_FILE.parent.mkdir(parents=True, exist_ok=True)
            COST_FILE.write_text(
                json.dumps(self.to_dict(), indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            log.debug("Failed to save session costs: %s", e)

    @classmethod
    def load(cls, session_id: str = "") -> "SessionCosts":
        """Load previous session costs if session_id matches."""
        if not COST_FILE.exists():
            return cls(session_id=session_id, started_at=time.time())
        try:
            data = json.loads(COST_FILE.read_text(encoding="utf-8"))
            if data.get("session_id") == session_id and session_id:
                costs = cls(
                    session_id=session_id,
                    started_at=data.get("started_at", time.time()),
                    total_cost_usd=data.get("total_cost_usd", 0),
                    total_calls=data.get("total_calls", 0),
                    total_errors=data.get("total_errors", 0),
                )
                for key, usage_data in data.get("model_usage", {}).items():
                    costs.model_usage[key] = ModelUsage(**usage_data)
                return costs
        except Exception as e:
            log.debug("Failed to load session costs: %s", e)
        return cls(session_id=session_id, started_at=time.time())


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a call based on known pricing."""
    pricing = _PRICING.get(model)
    if not pricing:
        # Try partial match (model name may have version suffix)
        for known_model, price in _PRICING.items():
            if known_model in model or model in known_model:
                pricing = price
                break
    if not pricing:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_session: SessionCosts | None = None
_session_lock = threading.Lock()


def get_session_costs(session_id: str = "") -> SessionCosts:
    """Get or create the session-level cost tracker (thread-safe singleton)."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = SessionCosts(session_id=session_id, started_at=time.time())
    return _session


def _save_on_exit() -> None:
    if _session is not None:
        _session.save()
        log.info("Session costs saved: %s", _session.format_summary())


atexit.register(_save_on_exit)
