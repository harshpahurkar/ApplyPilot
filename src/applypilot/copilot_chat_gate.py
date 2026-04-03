"""Global + lane-aware Copilot chat gate.

Provides two layers of throttling for Copilot chat sessions:
1) A global concurrency cap across all Copilot chat calls.
2) Optional per-lane caps (scoring, tailoring, apply, etc.).

This lets the pipeline reserve one chat lane for scoring and one for
tailoring while still keeping a hard total cap to avoid usage spikes.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator


_state_lock = threading.Lock()
_global_sem: threading.BoundedSemaphore | None = None
_global_slots = 0
_lane_sems: dict[str, threading.BoundedSemaphore] = {}
_lane_slots: dict[str, int] = {}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def gate_enabled() -> bool:
    """Return whether the global single-chat gate is enabled."""
    return _truthy(os.environ.get("APPLYPILOT_COPILOT_GLOBAL_SINGLE_CHAT", "1"))


def _bounded_slots(raw: str | None, default: int) -> int:
    """Parse and clamp chat-slot counts to a safe range."""
    try:
        slots = int((raw or "").strip())
    except ValueError:
        slots = default
    return max(1, min(8, slots))


def _desired_global_slots() -> int:
    """Configured total concurrent Copilot chat slots."""
    return _bounded_slots(os.environ.get("APPLYPILOT_COPILOT_GLOBAL_CHAT_SLOTS", "1"), 1)


def _lane_from_owner(owner: str) -> str:
    """Map owner strings to lane keys."""
    o = (owner or "").strip().lower()
    if o.startswith("llm:scoring"):
        return "llm_scoring"
    if o.startswith("llm:tailoring"):
        return "llm_tailoring"
    if o.startswith("llm:"):
        return "llm_general"
    if o.startswith("apply:"):
        return "apply"
    return "default"


def _desired_lane_slots(lane: str) -> int:
    """Configured slots for a specific lane."""
    env_map = {
        "llm_scoring": "APPLYPILOT_COPILOT_SCORING_CHAT_SLOTS",
        "llm_tailoring": "APPLYPILOT_COPILOT_TAILORING_CHAT_SLOTS",
        "llm_general": "APPLYPILOT_COPILOT_GENERAL_CHAT_SLOTS",
        "apply": "APPLYPILOT_COPILOT_APPLY_CHAT_SLOTS",
        "default": "APPLYPILOT_COPILOT_DEFAULT_CHAT_SLOTS",
    }
    defaults = {
        "llm_scoring": 1,
        "llm_tailoring": 1,
        "llm_general": 1,
        "apply": 1,
        "default": 1,
    }
    env_name = env_map.get(lane, "APPLYPILOT_COPILOT_DEFAULT_CHAT_SLOTS")
    default_slots = defaults.get(lane, 1)
    return _bounded_slots(os.environ.get(env_name, str(default_slots)), default_slots)


def _get_global_semaphore() -> threading.BoundedSemaphore:
    """Get (or initialize) the global semaphore for current slot config."""
    global _global_sem, _global_slots
    slots = _desired_global_slots()
    with _state_lock:
        if _global_sem is None or _global_slots != slots:
            _global_sem = threading.BoundedSemaphore(slots)
            _global_slots = slots
        return _global_sem


def _get_lane_semaphore(lane: str) -> threading.BoundedSemaphore:
    """Get (or initialize) the semaphore for a specific lane."""
    slots = _desired_lane_slots(lane)
    with _state_lock:
        if lane not in _lane_sems or _lane_slots.get(lane) != slots:
            _lane_sems[lane] = threading.BoundedSemaphore(slots)
            _lane_slots[lane] = slots
        return _lane_sems[lane]


@contextmanager
def copilot_chat_slot(_owner: str) -> Iterator[float]:
    """Acquire the global Copilot chat slot.

    Yields:
        Seconds spent waiting for the slot.
    """
    if not gate_enabled():
        yield 0.0
        return

    lane = _lane_from_owner(_owner)
    global_gate = _get_global_semaphore()
    lane_gate = _get_lane_semaphore(lane)

    start = time.time()
    # Always acquire in deterministic order to avoid deadlocks.
    global_gate.acquire()
    lane_gate.acquire()
    waited = max(0.0, time.time() - start)
    try:
        yield waited
    finally:
        lane_gate.release()
        global_gate.release()
