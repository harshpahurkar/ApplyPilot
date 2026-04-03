"""
Unified LLM client for ApplyPilot — multi-provider with automatic fallback.

Loads ALL available providers from environment and tries them in order.
If the primary provider is rate-limited (429) or down (503/timeout),
the request automatically falls through to the next provider.

Priority order:
    0. Copilot CLI (optional, APPLYPILOT_USE_COPILOT_LLM=1)
  1. GEMINI_API_KEY  -> Google Gemini (default: gemini-2.0-flash) — FREE tier
  2. DEEPSEEK_API_KEY -> DeepSeek (default: deepseek-chat) — cheap fallback
  3. OPENAI_API_KEY  -> OpenAI (default: gpt-4o)
  4. LLM_URL         -> Local llama.cpp / Ollama compatible endpoint

LLM_MODEL env var overrides the model name for the PRIMARY provider only.
"""

import json
import logging
import os
import platform
import random
import re
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field

import httpx
from applypilot.copilot_chat_gate import copilot_chat_slot
from applypilot.cost_tracker import get_session_costs
from applypilot.apply.stealth import (
    get_pacer, get_budget, prepare_copilot_env,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection — build ALL available providers
# ---------------------------------------------------------------------------

_GEMINI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _copilot_base_command() -> list[str] | None:
    """Resolve the executable command prefix for Copilot CLI invocations."""
    custom = os.environ.get("APPLYPILOT_COPILOT_LLM_CMD", "").strip()
    if custom:
        return shlex.split(custom, posix=(platform.system() != "Windows"))

    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA", "")
        node_exe = shutil.which("node")
        npm_loader = Path(appdata) / "npm" / "node_modules" / "@github" / "copilot" / "npm-loader.js"
        if node_exe and npm_loader.exists():
            return [node_exe, str(npm_loader)]

    copilot_exe = shutil.which("copilot")
    if copilot_exe:
        return [copilot_exe]

    gh_exe = shutil.which("gh")
    if gh_exe:
        return [gh_exe, "copilot", "--"]

    return None


def _normalize_copilot_model(model: str) -> str:
    """Map user-facing Claude-style aliases to Copilot-supported model names."""
    requested = (model or "").strip()
    default_model = (
        os.environ.get("APPLYPILOT_COPILOT_LLM_DEFAULT_MODEL", "").strip()
        or "gpt-4.1"
    )

    if not requested:
        return default_model

    lowered = requested.lower()
    haiku_aliases = {
        "haiku",
        "claude-haiku",
        "claude_haiku",
        "claude-3-haiku",
        "claude-3.5-haiku",
        "claude-3-5-haiku",
        "claude-3-5-haiku-latest",
    }
    if lowered in haiku_aliases:
        mapped = (
            os.environ.get("APPLYPILOT_COPILOT_LLM_HAIKU_EQUIVALENT", "").strip()
            or default_model
        )
        if mapped != requested:
            log.info("Mapped Copilot LLM model '%s' to '%s'", requested, mapped)
        return mapped

    return requested


def _build_providers() -> list[ProviderConfig]:
    """Discover all configured LLM providers from environment variables.

    Returns providers in priority order, with optional Copilot first:
    Copilot (optional) -> Gemini -> DeepSeek -> OpenAI -> Local.
    """
    model_override = os.environ.get("LLM_MODEL", "")
    providers: list[ProviderConfig] = []

    use_copilot = _truthy(os.environ.get("APPLYPILOT_USE_COPILOT_LLM"))
    if use_copilot:
        copilot_cmd = _copilot_base_command()
        if copilot_cmd:
            configured_model = os.environ.get("COPILOT_LLM_MODEL", "haiku")
            providers.append(ProviderConfig(
                name="copilot",
                base_url="",
                model=_normalize_copilot_model(configured_model),
                api_key="",
            ))
        else:
            log.warning(
                "APPLYPILOT_USE_COPILOT_LLM=1 but Copilot CLI was not found; skipping Copilot provider"
            )

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    local_url = os.environ.get("LLM_URL", "")

    if gemini_key:
        providers.append(ProviderConfig(
            name="gemini",
            base_url=_GEMINI_COMPAT_BASE,
            model=model_override or "gemini-2.0-flash",
            api_key=gemini_key,
        ))


    if deepseek_key:
        providers.append(ProviderConfig(
            name="deepseek",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            api_key=deepseek_key,
        ))

    if openai_key:
        providers.append(ProviderConfig(
            name="openai",
            base_url="https://api.openai.com/v1",
            model=model_override or "gpt-4o",
            api_key=openai_key,
        ))

    if local_url:
        # LOCAL_LLM_MODEL lets you pin a specific Ollama/llama.cpp model
        # independently of LLM_MODEL (which only overrides the primary provider).
        local_model = os.environ.get("LOCAL_LLM_MODEL") or model_override or "local-model"
        providers.append(ProviderConfig(
            name="local",
            base_url=local_url.rstrip("/"),
            model=local_model,
            api_key=os.environ.get("LLM_API_KEY", ""),
        ))

    if not providers:
        raise RuntimeError(
            "No LLM provider configured. "
            "Set APPLYPILOT_USE_COPILOT_LLM=1, GEMINI_API_KEY, "
            "DEEPSEEK_API_KEY, OPENAI_API_KEY, or LLM_URL."
        )

    return providers


# ---------------------------------------------------------------------------
# Gemini native API helpers
# ---------------------------------------------------------------------------

class _GeminiCompatForbidden(Exception):
    """Sentinel: Gemini OpenAI-compat returned 403. Switch to native API."""
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Gemini compat 403: {response.text[:200]}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3   # retries per provider
_COPILOT_MAX_RETRIES = 2  # fewer retries for Copilot — fail fast, fall to Gemini
_TIMEOUT = 300     # seconds (5 min — Ollama needs time for 8K token generation)

# Channel-aware Copilot CLI timeouts (seconds).  Tailoring prompts are
# large and Sonnet 4.6 can take 3-4 min to generate a full resume, so
# tailoring gets the full 300s.  Scoring is fast (small prompt).
_COPILOT_TIMEOUT_DEFAULTS: dict[str, int] = {
    "scoring": 120,
    "tailoring": 600,
    "general": 150,
}

# Gemini free tier is 15 RPM = 4s minimum between requests; 10s gives headroom.
_RATE_LIMIT_BASE_WAIT = 10

# Consecutive failure tracking: after this many consecutive failures
# per provider, disable the provider for the rest of the session.
_MAX_CONSECUTIVE_FAILURES = int(os.environ.get("APPLYPILOT_LLM_MAX_CONSECUTIVE_FAILURES", "5"))

# Network-level errors that are safe to retry (connection drops, etc.)
_RETRYABLE_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ConnectError,
)


def _jittered_backoff(base: float, attempt: int, cap: float = 120.0) -> float:
    """Exponential backoff with 0-25% random jitter (prevents thundering herd)."""
    delay = min(base * (2 ** attempt), cap)
    jitter = delay * random.uniform(0, 0.25)
    return delay + jitter


def _copilot_channel_from_messages(messages: list[dict]) -> str:
    """Infer Copilot LLM lane from prompt content."""
    system_blob = "\n".join(
        str(m.get("content", ""))
        for m in messages
        if str(m.get("role", "")).lower() == "system"
    ).lower()

    if "ats-aware job fit evaluator" in system_blob or "respond in exactly this format" in system_blob and "score:" in system_blob:
        return "scoring"
    if "elite ats optimization engine" in system_blob or "resume tailoring" in system_blob:
        return "tailoring"
    if "resume quality judge" in system_blob:
        return "tailoring"
    if "cover letter" in system_blob:
        return "tailoring"
    return "general"


class LLMClient:
    """Multi-provider chat completions client with Gemini native API support.

    Tries each provider in order. If a provider is rate-limited (429),
    unavailable (503), or times out, the request falls through to the
    next provider before giving up.

    For Gemini: starts on the OpenAI-compat layer. On a 403 (which
    happens with preview/experimental models not exposed via compat),
    automatically switches to native generateContent API for the rest
    of the process.
    """

    def __init__(self, providers: list[ProviderConfig]) -> None:
        self.providers = providers
        self._client = httpx.Client(timeout=_TIMEOUT)
        # Track whether we've switched to native Gemini API
        self._use_native_gemini: bool = False
        # Circuit breaker: skip provider until cooldown_until timestamp
        self._cooldown_until: dict[str, float] = {}
        # Consecutive failure tracking: counts reset on success, provider
        # disabled for the session when _MAX_CONSECUTIVE_FAILURES reached.
        self._consecutive_failures: dict[str, int] = {p.name: 0 for p in providers}
        self._session_disabled: dict[str, bool] = {p.name: False for p in providers}
        # Per-provider mutex: serializes concurrent requests so only one thread
        # hits a given provider at a time (prevents simultaneous 429 storms).
        # Copilot serialization defaults ON for usage-cap safety.
        self._copilot_serialize = _truthy(os.environ.get("APPLYPILOT_COPILOT_LLM_SERIALIZE", "1"))
        self._provider_locks: dict[str, threading.Lock] = {
            p.name: threading.Lock()
            for p in providers
            if p.name != "copilot" or self._copilot_serialize
        }
        self._copilot_throttle_lock = threading.Lock()
        self._copilot_last_request_ts = 0.0
        self._copilot_min_interval = max(
            0.0,
            _safe_float_env("APPLYPILOT_COPILOT_LLM_MIN_INTERVAL", 0.0),
        )
        self._copilot_retry_base_wait = max(
            0.0,
            _safe_float_env("APPLYPILOT_COPILOT_LLM_RETRY_BASE_WAIT", 15.0),
        )
        self._max_cooldown_wait = max(
            0.0,
            _safe_float_env("APPLYPILOT_LLM_MAX_COOLDOWN_WAIT", 0.0),
        )
        self._copilot_chat_mode = os.environ.get("APPLYPILOT_COPILOT_LLM_CHAT_MODE", "single").strip().lower()
        if self._copilot_chat_mode not in {"single", "dual", "stateless"}:
            self._copilot_chat_mode = "single"

        self._copilot_resume_id = os.environ.get("APPLYPILOT_COPILOT_LLM_RESUME_ID", "").strip()
        self._copilot_resume_ids = {
            "scoring": os.environ.get("APPLYPILOT_COPILOT_LLM_RESUME_ID_SCORING", "").strip(),
            "tailoring": os.environ.get("APPLYPILOT_COPILOT_LLM_RESUME_ID_TAILORING", "").strip(),
            "general": os.environ.get("APPLYPILOT_COPILOT_LLM_RESUME_ID_GENERAL", "").strip(),
        }
        self._copilot_continue = (
            _truthy(os.environ.get("APPLYPILOT_COPILOT_LLM_CONTINUE", "0"))
            and not self._copilot_resume_id
        )
        names = [p.name for p in providers]
        log.info("LLM providers loaded: %s (primary: %s)", names, names[0])

    def _copilot_session_args_for_channel(self, channel: str) -> list[str]:
        """Build --resume/--continue args for a specific Copilot lane."""
        channel_key = channel if channel in self._copilot_resume_ids else "general"
        resume_id = self._copilot_resume_ids.get(channel_key, "")
        if resume_id:
            return ["--resume", resume_id]

        if self._copilot_resume_id:
            return ["--resume", self._copilot_resume_id]

        if self._copilot_chat_mode == "stateless":
            return []

        if self._copilot_chat_mode == "dual":
            # Dual mode avoids implicit --continue because Copilot CLI continue
            # is global and can blend scoring/tailoring context unexpectedly.
            return []

        return ["--continue"] if self._copilot_continue else []

    def _copilot_rate_limit_cooldown_seconds(self, error_text: str) -> float:
        """Infer cooldown from rate-limit messages, with a safe fallback."""
        text = (error_text or "").lower()
        m_hours = re.search(r"try again in\s*(\d+)\s*hour", text)
        if m_hours:
            return float(max(1, int(m_hours.group(1))) * 3600)
        m_mins = re.search(r"try again in\s*(\d+)\s*min", text)
        if m_mins:
            return float(max(1, int(m_mins.group(1))) * 60)
        return max(30.0, _safe_float_env("APPLYPILOT_COPILOT_LLM_RATE_LIMIT_COOLDOWN", 7200.0))

    def _throttle_copilot_llm(self) -> None:
        """Pace Copilot LLM calls to reduce burst-based rate-limit hits."""
        if self._copilot_min_interval <= 0:
            return
        with self._copilot_throttle_lock:
            now = time.time()
            elapsed = now - self._copilot_last_request_ts
            wait_s = max(0.0, self._copilot_min_interval - elapsed)
            if wait_s > 0:
                time.sleep(wait_s)
            self._copilot_last_request_ts = time.time()

    def _record_failure(self, provider_name: str) -> None:
        """Increment consecutive failure counter; disable provider if threshold reached."""
        self._consecutive_failures[provider_name] = self._consecutive_failures.get(provider_name, 0) + 1
        count = self._consecutive_failures[provider_name]
        if count >= _MAX_CONSECUTIVE_FAILURES and not self._session_disabled.get(provider_name, False):
            self._session_disabled[provider_name] = True
            log.warning(
                "[%s] disabled for session after %d consecutive failures",
                provider_name, count,
            )

    def _render_messages_for_copilot(self, messages: list[dict]) -> str:
        """Render OpenAI-style chat messages into a plain prompt string."""
        sections: list[str] = [
            "You are an expert resume tailoring and job-scoring assistant.",
            "Follow the SYSTEM instructions strictly.",
            "Return only the requested output format with no extra framing.",
            "",
        ]
        for msg in messages:
            role = str(msg.get("role", "user")).upper()
            content = str(msg.get("content", "")).strip()
            sections.append(f"[{role}]\n{content}\n")
        rendered = "\n".join(sections).strip()

        max_chars = int(os.environ.get("APPLYPILOT_COPILOT_LLM_PROMPT_MAX", "24000"))
        if len(rendered) > max_chars:
            head = int(max_chars * 0.7)
            tail = max_chars - head
            rendered = (
                rendered[:head]
                + "\n\n[Prompt truncated for Copilot LLM limits]\n\n"
                + rendered[-tail:]
            )
        return rendered

    def _extract_copilot_output_text(self, output: str) -> str:
        """Parse Copilot JSON output and return the final assistant text."""
        chunks: list[str] = []

        def _maybe_add(obj: dict) -> None:
            msg_type = str(obj.get("type", ""))
            if msg_type == "assistant.message":
                data = obj.get("data") or {}
                text = data.get("message") or data.get("content")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
            for key in ("content", "message", "text"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    chunks.append(val.strip())

        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                _maybe_add(parsed)

        if not chunks:
            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    _maybe_add(parsed)
            except json.JSONDecodeError:
                pass

        if chunks:
            return chunks[-1]

        # Fallback: plain-text output (rare, but possible if CLI formatting changes)
        return output.strip()

    def _chat_copilot(
        self,
        provider: ProviderConfig,
        messages: list[dict],
        _temperature: float,
        _max_tokens: int,
    ) -> str:
        """Call Copilot CLI for non-browser LLM tasks (scoring/tailoring/cover)."""
        base_cmd = _copilot_base_command()
        if not base_cmd:
            raise RuntimeError("Copilot CLI is not available on PATH")

        prompt = self._render_messages_for_copilot(messages)
        channel = _copilot_channel_from_messages(messages)
        # Channel-aware timeout: scoring is snappy, tailoring needs more
        env_key = f"APPLYPILOT_COPILOT_LLM_TIMEOUT_{channel.upper()}"
        channel_default = _COPILOT_TIMEOUT_DEFAULTS.get(channel, 120)
        timeout_s = int(os.environ.get(env_key, os.environ.get("APPLYPILOT_COPILOT_LLM_TIMEOUT", str(channel_default))))

        cmd = [
            *base_cmd,
            "--model", _normalize_copilot_model(provider.model),
            "--no-ask-user",
            "--output-format", "json",
        ]
        cmd.extend(self._copilot_session_args_for_channel(channel))

        env = prepare_copilot_env()

        log.info("[copilot] calling %s channel=%s prompt=%d chars timeout=%ds",
                 provider.model, channel, len(prompt), timeout_s)
        # Pass prompt via stdin to avoid Windows command-line length limits
        # (>~24K chars via -p causes subprocess.run to hang on Windows).
        with copilot_chat_slot(f"llm:{channel}:{provider.model}") as gate_wait_s:
            if gate_wait_s > 0.1:
                log.info("[copilot] global chat gate wait %.1fs", gate_wait_s)
            t0 = time.time()
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                env=env,
                input=prompt,
            )
        elapsed = time.time() - t0
        log.info("[copilot] response in %.1fs, exit=%d, stdout=%d stderr=%d",
                 elapsed, proc.returncode, len(proc.stdout or ""), len(proc.stderr or ""))
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

        if proc.returncode != 0:
            snippet = output.strip()[-400:]
            raise RuntimeError(f"Copilot CLI exited {proc.returncode}: {snippet}")

        text = self._extract_copilot_output_text(output)
        if not text:
            raise RuntimeError("Copilot CLI returned empty output")
        return text

    # -- Native Gemini API --------------------------------------------------

    def _chat_native_gemini(
        self,
        provider: ProviderConfig,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the native Gemini generateContent API.

        Converts OpenAI-style messages to Gemini's contents/systemInstruction
        format transparently.
        """
        contents: list[dict] = []
        system_parts: list[dict] = []

        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"{_GEMINI_NATIVE_BASE}/models/{provider.model}:generateContent"
        resp = self._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": provider.api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    # -- OpenAI-compat API --------------------------------------------------

    def _chat_compat(
        self,
        provider: ProviderConfig,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the OpenAI-compatible endpoint."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

        payload = {
            "model": provider.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = self._client.post(
            f"{provider.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )

        # 403 on Gemini compat = model not available on compat layer
        if resp.status_code == 403 and provider.name == "gemini":
            raise _GeminiCompatForbidden(resp)

        return resp, resp.json() if resp.status_code == 200 else (resp, None)

    # -- internal -----------------------------------------------------------

    def _try_provider(
        self,
        provider: ProviderConfig,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str | None:
        """Try a single provider with retries. Returns response text or None.

        Uses a per-provider mutex so that when multiple threads all try the
        same provider simultaneously only one fires at a time — this prevents
        all workers hitting Gemini at once and triggering simultaneous 429s.
        """
        lock = self._provider_locks.get(provider.name)
        if lock:
            lock.acquire()
        try:
            return self._try_provider_locked(provider, messages, temperature, max_tokens)
        finally:
            if lock:
                lock.release()

    def _try_provider_locked(
        self,
        provider: ProviderConfig,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str | None:
        """Inner implementation of _try_provider (called under the provider lock)."""
        # Circuit breaker: skip this provider if it's in cooldown.
        cooldown_until = self._cooldown_until.get(provider.name, 0)
        if time.time() < cooldown_until:
            remaining = int(cooldown_until - time.time())
            log.debug("[%s] in cooldown for %ds, skipping", provider.name, remaining)
            return None

        # Session-level disable: after too many consecutive failures
        if self._session_disabled.get(provider.name, False):
            log.debug("[%s] session-disabled after consecutive failures, skipping", provider.name)
            return None

        if provider.name == "copilot":
            for attempt in range(_COPILOT_MAX_RETRIES):
                try:
                    self._throttle_copilot_llm()
                    # Stealth: human-like inter-LLM delay + budget tracking
                    pacer = get_pacer()
                    pacer.sleep_inter_llm()
                    budget = get_budget()
                    budget.wait_if_exhausted()
                    content = self._chat_copilot(provider, messages, temperature, max_tokens)
                    budget.record()
                    # Success — reset cooldown, failure counter, and pacer penalty
                    self._cooldown_until.pop(provider.name, None)
                    self._consecutive_failures[provider.name] = 0
                    pacer.report_success()
                    log.debug("[%s] response OK (%d chars)", provider.name, len(content))
                    # Copilot doesn't expose token counts — record call only
                    get_session_costs().record_call(
                        provider=provider.name, model=provider.model,
                    )
                    return content
                except subprocess.TimeoutExpired:
                    self._record_failure(provider.name)
                    pacer.report_timeout()  # adaptive backoff
                    if attempt < _COPILOT_MAX_RETRIES - 1:
                        wait = _jittered_backoff(self._copilot_retry_base_wait, attempt)
                        log.warning("[%s] timeout, retrying in %.1fs", provider.name, wait)
                        if wait > 0:
                            time.sleep(wait)
                        continue
                    self._cooldown_until[provider.name] = time.time() + 120
                    log.warning("[%s] timeout after retries, cooling down 120s", provider.name)
                    return None
                except Exception as e:
                    msg = str(e).lower()
                    if "429" in msg or "rate limit" in msg or "(rate_limit)" in msg:
                        self._record_failure(provider.name)
                        pacer.report_timeout()  # treat rate-limit as timeout for pacing
                        cooldown = self._copilot_rate_limit_cooldown_seconds(str(e))
                        self._cooldown_until[provider.name] = time.time() + cooldown
                        log.warning("[%s] rate-limited, cooling down %.0fs", provider.name, cooldown)
                        return None
                    self._record_failure(provider.name)
                    if attempt < _COPILOT_MAX_RETRIES - 1:
                        wait = _jittered_backoff(self._copilot_retry_base_wait, attempt)
                        log.warning("[%s] %s, retrying in %.1fs", provider.name, type(e).__name__, wait)
                        if wait > 0:
                            time.sleep(wait)
                        continue
                    log.warning("[%s] %s, falling back...", provider.name, str(e)[:200])
                    return None

        is_gemini = provider.name == "gemini"

        # Qwen3 optimization
        if "qwen" in provider.model.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

        for attempt in range(_MAX_RETRIES):
            try:
                # Route to native Gemini if we've already confirmed it's needed
                if is_gemini and self._use_native_gemini:
                    content = self._chat_native_gemini(
                        provider, messages, temperature, max_tokens,
                    )
                    log.debug("[%s/native] response OK (%d chars)", provider.name, len(content))
                    self._consecutive_failures[provider.name] = 0
                    get_session_costs().record_call(
                        provider=provider.name, model=provider.model,
                    )
                    return content

                # Standard OpenAI-compat path
                headers: dict[str, str] = {"Content-Type": "application/json"}
                if provider.api_key:
                    headers["Authorization"] = f"Bearer {provider.api_key}"

                payload = {
                    "model": provider.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                resp = self._client.post(
                    f"{provider.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )

                # Gemini compat 403 → switch to native API
                if resp.status_code == 403 and is_gemini:
                    log.warning(
                        "Gemini compat endpoint returned 403 for model '%s'. "
                        "Switching to native generateContent API.",
                        provider.model,
                    )
                    self._use_native_gemini = True
                    try:
                        content = self._chat_native_gemini(
                            provider, messages, temperature, max_tokens,
                        )
                        log.debug("[%s/native] response OK (%d chars)", provider.name, len(content))
                        return content
                    except httpx.HTTPStatusError as native_exc:
                        log.error(
                            "Both Gemini endpoints failed. Compat: 403. Native: %s",
                            native_exc.response.status_code,
                        )
                        return None

                if resp.status_code in (429, 503):
                    self._record_failure(provider.name)
                    if attempt < _MAX_RETRIES - 1:
                        # Respect Retry-After header (Gemini sends this)
                        retry_after = (
                            resp.headers.get("Retry-After")
                            or resp.headers.get("X-RateLimit-Reset-Requests")
                        )
                        if retry_after:
                            try:
                                wait = float(retry_after)
                            except (ValueError, TypeError):
                                wait = _jittered_backoff(_RATE_LIMIT_BASE_WAIT, attempt, cap=60)
                        elif is_gemini:
                            wait = _jittered_backoff(_RATE_LIMIT_BASE_WAIT, attempt, cap=60)
                        else:
                            wait = _jittered_backoff(1, attempt, cap=60)
                        log.warning(
                            "[%s] %s, retrying in %.1fs (attempt %d/%d)",
                            provider.name, resp.status_code, wait,
                            attempt + 1, _MAX_RETRIES,
                        )
                        time.sleep(wait)
                        continue
                    # Circuit breaker: put provider in cooldown so next requests
                    # skip straight to fallback without waiting again
                    cooldown = 120 if is_gemini else 30
                    self._cooldown_until[provider.name] = time.time() + cooldown
                    log.warning(
                        "[%s] rate-limited (status: %s), cooling down %ds, falling back...",
                        provider.name, resp.status_code, cooldown,
                    )
                    return None

                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                log.debug("[%s] response OK (%d chars)", provider.name, len(content))
                # Successful response — clear any cooldown and reset failure counter
                self._cooldown_until.pop(provider.name, None)
                self._consecutive_failures[provider.name] = 0
                # Track cost
                usage = data.get("usage", {})
                get_session_costs().record_call(
                    provider=provider.name,
                    model=provider.model,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )
                return content

            except _RETRYABLE_NETWORK_ERRORS as e:
                self._record_failure(provider.name)
                if attempt < _MAX_RETRIES - 1:
                    wait = _jittered_backoff(_RATE_LIMIT_BASE_WAIT if is_gemini else 1, attempt, cap=60)
                    log.warning(
                        "[%s] %s, retrying in %.1fs (attempt %d/%d)",
                        provider.name, type(e).__name__, wait,
                        attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                log.warning("[%s] %s after %d attempts, falling back...",
                            provider.name, type(e).__name__, _MAX_RETRIES)
                return None

            except httpx.HTTPStatusError as e:
                self._record_failure(provider.name)
                sc = e.response.status_code
                if sc == 402:
                    # Payment required — no credits. Put in 24h cooldown.
                    self._cooldown_until[provider.name] = time.time() + 86400
                    log.error("[%s] HTTP 402: no credits, disabling for this session", provider.name)
                else:
                    log.error("[%s] HTTP %s: %s", provider.name, sc, str(e)[:200])
                return None

        return None

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        _truncation_retry: bool = False,
    ) -> str:
        """Send a chat completion request, falling back across providers on failure.

        Transparent mid-call fallback: if the primary provider is rate-limited
        or fails, the request automatically falls through to the next provider.
        If ALL providers are in cooldown, waits for the soonest to recover
        rather than failing immediately (avoids silent data loss).

        Reactive prompt truncation: if the prompt is rejected as too long by
        all providers, the longest user message is trimmed by 30% and retried
        once.
        """
        for provider in self.providers:
            result = self._try_provider(provider, messages, temperature, max_tokens)
            if result is not None:
                return result

        # All providers failed or in cooldown.  Wait for the soonest cooldown
        # to expire and retry — use _max_cooldown_wait if set, otherwise cap
        # at 5 minutes (300s) to avoid blocking forever on long cooldowns.
        now = time.time()
        max_wait = self._max_cooldown_wait if self._max_cooldown_wait > 0 else 300.0
        candidates = [
            (self._cooldown_until.get(p.name, 0), p)
            for p in self.providers
            if not self._session_disabled.get(p.name, False)
            and self._cooldown_until.get(p.name, 0) > 0
            and self._cooldown_until.get(p.name, 0) < now + max_wait
        ]
        if candidates:
            # Sort by soonest wake time — try up to 2 providers
            candidates.sort(key=lambda x: x[0])
            for wake_at, wake_provider in candidates[:2]:
                wait = max(0.0, wake_at - now)
                if wait > 0:
                    log.warning(
                        "All providers exhausted — waiting %.0fs for %s to recover",
                        wait, wake_provider.name,
                    )
                    time.sleep(wait)
                self._cooldown_until.pop(wake_provider.name, None)
                result = self._try_provider(wake_provider, messages, temperature, max_tokens)
                if result is not None:
                    return result
                now = time.time()  # refresh after wait+retry

        # Reactive prompt truncation: if this wasn't already a truncation retry,
        # trim the longest user message by 30% and try all providers once more.
        if not _truncation_retry:
            truncated = self._truncate_messages(messages)
            if truncated is not None:
                log.warning("Retrying with truncated prompt (30%% shorter)")
                try:
                    return self.chat(
                        truncated, temperature, max_tokens,
                        _truncation_retry=True,
                    )
                except RuntimeError:
                    pass  # truncation didn't help, fall through to original error

        raise RuntimeError(
            f"All LLM providers failed ({', '.join(p.name for p in self.providers)}). "
            "Check API keys, quotas, and network connectivity."
        )

    @staticmethod
    def _truncate_messages(messages: list[dict], fraction: float = 0.3) -> list[dict] | None:
        """Trim the longest user message by `fraction` to fit context limits.

        Returns a new message list or None if no user messages to truncate.
        """
        longest_idx = -1
        longest_len = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                content_len = len(msg.get("content", ""))
                if content_len > longest_len:
                    longest_len = content_len
                    longest_idx = i

        if longest_idx < 0 or longest_len < 200:
            return None  # nothing to trim

        trimmed = list(messages)
        content = trimmed[longest_idx]["content"]
        keep = int(longest_len * (1 - fraction))
        # Keep more from the beginning (likely contains instructions)
        head = int(keep * 0.7)
        tail = keep - head
        trimmed[longest_idx] = {
            **trimmed[longest_idx],
            "content": (
                content[:head]
                + "\n\n[Content truncated to fit model context window]\n\n"
                + content[-tail:]
            ),
        }
        return trimmed

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: LLMClient | None = None
_instance_lock = threading.Lock()


def get_client() -> LLMClient:
    """Return (or create) the module-level LLMClient singleton (thread-safe)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                providers = _build_providers()
                _instance = LLMClient(providers)
    return _instance
