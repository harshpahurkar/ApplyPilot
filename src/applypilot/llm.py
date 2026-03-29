"""
Unified LLM client for ApplyPilot — multi-provider with automatic fallback.

Loads ALL available providers from environment and tries them in order.
If the primary provider is rate-limited (429) or down (503/timeout),
the request automatically falls through to the next provider.

Priority order:
  1. GEMINI_API_KEY  -> Google Gemini (default: gemini-2.0-flash) — FREE tier
  2. DEEPSEEK_API_KEY -> DeepSeek (default: deepseek-chat) — cheap fallback
  3. OPENAI_API_KEY  -> OpenAI (default: gpt-4o)
  4. LLM_URL         -> Local llama.cpp / Ollama compatible endpoint

LLM_MODEL env var overrides the model name for the PRIMARY provider only.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass

import httpx

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


def _build_providers() -> list[ProviderConfig]:
    """Discover all configured LLM providers from environment variables.

    Returns a list ordered by priority: Gemini (free) -> DeepSeek -> OpenAI -> Local.
    """
    model_override = os.environ.get("LLM_MODEL", "")
    providers: list[ProviderConfig] = []

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
            "Set GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, or LLM_URL."
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
_TIMEOUT = 300     # seconds (5 min — Ollama needs time for 8K token generation)

# Gemini free tier is 15 RPM = 4s minimum between requests; 10s gives headroom.
_RATE_LIMIT_BASE_WAIT = 10

# Network-level errors that are safe to retry (connection drops, etc.)
_RETRYABLE_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ConnectError,
)


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
        # Per-provider mutex: serializes concurrent requests so only one thread
        # hits a given provider at a time (prevents simultaneous 429 storms).
        self._provider_locks: dict[str, threading.Lock] = {
            p.name: threading.Lock() for p in providers
        }
        names = [p.name for p in providers]
        log.info("LLM providers loaded: %s (primary: %s)", names, names[0])

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
        is_gemini = provider.name == "gemini"

        # Circuit breaker: skip this provider if it's in cooldown
        cooldown_until = self._cooldown_until.get(provider.name, 0)
        if time.time() < cooldown_until:
            remaining = int(cooldown_until - time.time())
            log.debug("[%s] in cooldown for %ds, skipping", provider.name, remaining)
            return None

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
                    # For Gemini: only 1 retry then immediate cooldown + fallback
                    # (free tier quota exhausts quickly; no point burning 130s waiting)
                    max_retries_here = 1 if is_gemini else _MAX_RETRIES
                    if attempt < max_retries_here - 1:
                        # Respect Retry-After header (Gemini sends this)
                        retry_after = (
                            resp.headers.get("Retry-After")
                            or resp.headers.get("X-RateLimit-Reset-Requests")
                        )
                        if retry_after:
                            try:
                                wait = float(retry_after)
                            except (ValueError, TypeError):
                                wait = _RATE_LIMIT_BASE_WAIT * (2 ** attempt)
                        elif is_gemini:
                            wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                        else:
                            wait = 2 ** attempt
                        log.warning(
                            "[%s] %s, retrying in %ds (attempt %d/%d)",
                            provider.name, resp.status_code, wait,
                            attempt + 1, max_retries_here,
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
                # Successful response — clear any cooldown for this provider
                self._cooldown_until.pop(provider.name, None)
                return content

            except _RETRYABLE_NETWORK_ERRORS as e:
                if attempt < _MAX_RETRIES - 1:
                    wait = _RATE_LIMIT_BASE_WAIT if is_gemini else 2 ** attempt
                    log.warning(
                        "[%s] %s, retrying in %ds (attempt %d/%d)",
                        provider.name, type(e).__name__, wait,
                        attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                log.warning("[%s] %s after %d attempts, falling back...",
                            provider.name, type(e).__name__, _MAX_RETRIES)
                return None

            except httpx.HTTPStatusError as e:
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
    ) -> str:
        """Send a chat completion request, falling back across providers on failure.

        If all providers are in cooldown, waits for the soonest to recover
        rather than failing immediately (avoids silent data loss when Gemini
        free-tier RPM is exhausted and there are no other providers).
        """
        for provider in self.providers:
            result = self._try_provider(provider, messages, temperature, max_tokens)
            if result is not None:
                return result

        # All providers failed or in cooldown. If any have a finite cooldown,
        # wait for the soonest one to recover and retry once.
        now = time.time()
        candidates = [
            (self._cooldown_until.get(p.name, 0), p)
            for p in self.providers
            if self._cooldown_until.get(p.name, 0) < now + 300  # max 5-min wait
        ]
        if candidates:
            wake_at, wake_provider = min(candidates, key=lambda x: x[0])
            wait = max(0.0, wake_at - now)
            if wait > 0:
                log.warning(
                    "[%s] all providers in cooldown — waiting %.0fs for %s to recover",
                    wake_provider.name, wait, wake_provider.name,
                )
                time.sleep(wait)
            # Clear the cooldown and retry
            self._cooldown_until.pop(wake_provider.name, None)
            result = self._try_provider(wake_provider, messages, temperature, max_tokens)
            if result is not None:
                return result

        raise RuntimeError(
            f"All LLM providers failed ({', '.join(p.name for p in self.providers)}). "
            "Check API keys, quotas, and network connectivity."
        )

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
