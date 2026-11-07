"""Unified LLM client with 3-key rotation, token and time tracking, and Gemini/Groq support (G3, Phase 2).

Features:
  - 3-key rotation ring: automatically rotates across multiple API keys on HTTP 429
    (rate limit) or quota thresholds, maximizing experiment throughput on free tiers.
  - Full Gemini API support: speaks the OpenAI-compatible chat completions interface
    (gemini-1.5-flash, gemini-2.0-flash, etc.) with automated key failover.
  - Detailed token and time tracking: logs prompt tokens, completion tokens, network latency,
    cache hits, retries, and key rotations for efficiency analysis (E6).
  - Optimizer vs Executor roles: builds specialized optimizer models (e.g., higher temperature
    and reasoning budget) or fast executor models.
  - Disk caching: re-running experiments reuses answers with zero API calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from src.config import load_dotenv


class QuotaExhausted(Exception):
    """Raised when all configured API keys for a provider have hit quota or rate limits."""
    pass


PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
    "ollama": ("http://localhost:11434/v1/chat/completions", None),
}

DEFAULT_KEY_ENVS = {
    "groq": ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4", "GROQ_API_KEY_5", "GROQ_API_KEY_6"],
    "gemini": ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GOOGLE_API_KEY"],
    "ollama": [],
}


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


@dataclass
class ClientStats:
    """Cumulative usage and efficiency tracking counters (E6)."""
    total_calls: int = 0
    cached_calls: int = 0
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_s: float = 0.0
    key_rotations: int = 0
    retries: int = 0
    rate_limit_hits: int = 0
    keys_used: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, int | float]:
        """Numeric summary for differential tracking (before/after deltas)."""
        return {
            "total_calls": self.total_calls,
            "cached_calls": self.cached_calls,
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_s": round(self.total_latency_s, 3),
            "key_rotations": self.key_rotations,
            "retries": self.retries,
            "rate_limit_hits": self.rate_limit_hits,
            "keys_used_count": len(self.keys_used),
        }


class LLMClient:
    def __init__(self, provider: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 768, requests_per_minute: int = 28,
                 timeout_s: int = 60, cache_path: str | None = None,
                 key_envs: list[str] | None = None, api_keys: list[str] | None = None):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Use one of {list(PROVIDERS)} or mock.")
        load_dotenv()
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens, self.timeout_s = temperature, max_tokens, timeout_s
        self.url, primary_key_env = PROVIDERS[provider]

        # Assemble the key rotation pool
        self.api_keys: list[str] = []
        if api_keys:
            self.api_keys = [k.strip() for k in api_keys if k and k.strip()]
        else:
            default_envs = DEFAULT_KEY_ENVS.get(provider, [primary_key_env] if primary_key_env else [])
            # Only use passed key_envs if they match this provider, otherwise fallback to provider defaults
            if key_envs and any(provider.upper() in k for k in key_envs):
                cand_envs = [k for k in key_envs if provider.upper() in k]
            else:
                cand_envs = default_envs

            for env_var in cand_envs:
                if not env_var:
                    continue
                val = os.environ.get(env_var)
                if val and val.strip() and val.strip() not in self.api_keys:
                    self.api_keys.append(val.strip())

        if primary_key_env and not self.api_keys:
            raise RuntimeError(
                f"No API key found for provider '{provider}'. Please set {primary_key_env} "
                f"(or {primary_key_env}_2, {primary_key_env}_3 for key rotation) in your .env file."
            )

        self._key_index = 0
        self.min_interval = 60.0 / max(requests_per_minute, 1)
        self._last_call = 0.0
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = self._load_cache()
        self.stats = ClientStats()

    @property
    def current_key(self) -> str | None:
        if not self.api_keys:
            return None
        return self.api_keys[self._key_index]

    def rotate_key(self, reason: str = "rate limit") -> str | None:
        """Rotate to the next available API key in the rotation ring."""
        if not self.api_keys or len(self.api_keys) <= 1:
            return self.current_key
        prev_idx = self._key_index
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        self.stats.key_rotations += 1
        masked_prev = f"{self.api_keys[prev_idx][:4]}...{self.api_keys[prev_idx][-3:]}"
        masked_new = f"{self.api_keys[self._key_index][:4]}...{self.api_keys[self._key_index][-3:]}"
        print(f"  [llm] Key rotation ({prev_idx + 1}->{self._key_index + 1}/{len(self.api_keys)}) "
              f"[{masked_prev} -> {masked_new}]: {reason}")
        return self.current_key

    # ---------- cache ----------
    def _key(self, messages: list[dict]) -> str:
        blob = json.dumps([self.provider, self.model, self.temperature, messages], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _load_cache(self) -> dict:
        cache = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        row = json.loads(line)
                        cache[row["key"]] = row
                    except json.JSONDecodeError:
                        continue
        return cache

    def _save(self, key: str, resp: LLMResponse) -> None:
        row = {"key": key, "text": resp.text, "prompt_tokens": resp.prompt_tokens,
               "completion_tokens": resp.completion_tokens, "latency_s": resp.latency_s}
        self._cache[key] = row
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    # ---------- call ----------
    def chat(self, messages: list[dict]) -> LLMResponse:
        self.stats.total_calls += 1
        key = self._key(messages)
        if key in self._cache:
            c = self._cache[key]
            self.stats.cached_calls += 1
            return LLMResponse(c["text"], c["prompt_tokens"], c["completion_tokens"],
                               c["latency_s"], cached=True)

        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}

        total_keys = max(1, len(self.api_keys))
        max_attempts = max(6, total_keys * 3)
        consecutive_rate_limits = 0

        for attempt in range(max_attempts):
            curr_key = self.current_key
            headers = {"Content-Type": "application/json"}
            url = self.url
            if curr_key:
                headers["Authorization"] = f"Bearer {curr_key}"
                self.stats.keys_used.add(curr_key)
                if self.provider == "gemini":
                    # Gemini OpenAI endpoint also accepts ?key= param
                    sep = "&" if "?" in url else "?"
                    url = f"{url}{sep}key={curr_key}"

            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            t0 = time.time()

            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout_s)
            except requests.RequestException as e:
                self.stats.retries += 1
                self._backoff(attempt, f"network error: {e}")
                continue

            lat = round(time.time() - t0, 3)

            # Check for Rate Limit or Server Errors
            if r.status_code == 429 or (r.status_code == 403 and "quota" in r.text.lower()):
                self.stats.rate_limit_hits += 1
                self.stats.retries += 1
                consecutive_rate_limits += 1
                retry_after = float(r.headers.get("retry-after", 0) or 0)

                # If we have multiple keys in rotation, rotate immediately
                if len(self.api_keys) > 1:
                    self.rotate_key(f"HTTP {r.status_code} quota/rate limit")
                    # If all keys failed consecutively in one cycle:
                    if consecutive_rate_limits >= len(self.api_keys):
                        if attempt >= max_attempts - 1:
                            raise QuotaExhausted(
                                f"All {len(self.api_keys)} API keys for {self.provider} exhausted daily quota."
                            )
                        self._backoff(attempt, f"All {len(self.api_keys)} keys rate-limited", retry_after)
                        consecutive_rate_limits = 0
                    continue

                if attempt >= max_attempts - 1:
                    raise QuotaExhausted(f"API key for {self.provider} exhausted quota or rate limits.")
                self._backoff(attempt, f"HTTP {r.status_code}", retry_after)
                continue

            if r.status_code >= 500:
                self.stats.retries += 1
                self._backoff(attempt, f"HTTP {r.status_code} provider error")
                continue

            if r.status_code != 200:
                raise RuntimeError(f"LLM call failed ({r.status_code}): {r.text[:300]}")

            consecutive_rate_limits = 0
            data = r.json()
            usage = data.get("usage") or {}
            p_tok = int(usage.get("prompt_tokens", 0))
            c_tok = int(usage.get("completion_tokens", 0))

            # Update live client stats
            self.stats.api_calls += 1
            self.stats.prompt_tokens += p_tok
            self.stats.completion_tokens += c_tok
            self.stats.total_tokens += (p_tok + c_tok)
            self.stats.total_latency_s += lat

            resp = LLMResponse(
                text=(data["choices"][0]["message"].get("content") or "").strip(),
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_s=lat,
            )
            self._save(key, resp)
            return resp

        raise QuotaExhausted(f"LLM call kept failing after {max_attempts} attempts (rate limit or outage).")

    @staticmethod
    def _backoff(attempt: int, reason: str, retry_after: float = 0) -> None:
        delay = max(retry_after, min(60, 2 ** attempt + random.random()))
        print(f"  [llm] {reason}; retrying in {delay:.1f}s")
        time.sleep(delay)


class MockLLM:
    """Offline stand-in: returns a deterministic pseudo-random valid answer.
    Lets you test the whole pipeline with no key and no internet.
    Its scores are meaningless; never report them."""

    provider, model = "mock", "mock"

    def __init__(self, **kwargs: Any):
        self.stats = ClientStats()

    def chat(self, messages: list[dict]) -> LLMResponse:
        self.stats.total_calls += 1
        self.stats.api_calls += 1
        prompt = messages[-1]["content"]
        rng = random.Random(hashlib.sha256(prompt.encode()).hexdigest())
        if "ANOMALOUS" in prompt:
            text = rng.choice(["NORMAL", "ANOMALOUS"])
        else:
            n = prompt.count("\n[")  # numbered options look like "[3] ..."
            text = str(rng.randint(1, max(n, 1)))

        p_tok = len(prompt) // 4
        c_tok = 1
        lat = 0.001
        self.stats.prompt_tokens += p_tok
        self.stats.completion_tokens += c_tok
        self.stats.total_tokens += (p_tok + c_tok)
        self.stats.total_latency_s += lat
        return LLMResponse(text, prompt_tokens=p_tok, completion_tokens=c_tok, latency_s=lat)


def make_llm(cfg: dict, provider: str | None = None, role: str = "executor"):
    """Instantiate an LLMClient or MockLLM based on configuration.

    Parameters
    ----------
    cfg : dict
        Experiment configuration loaded from config.yaml.
    provider : str | None
        Override provider ('groq', 'gemini', 'mock', 'ollama').
    role : str, default='executor'
        'executor' uses cfg['llm']; 'optimizer' uses cfg.get('optimizer_llm', {})
        for specialized proposer behavior (e.g., higher temperature / reasoning).
    """
    base_llm = dict(cfg.get("llm", {}))
    provider = provider or base_llm.get("provider", "groq")
    if provider == "mock":
        return MockLLM()

    if role == "optimizer" and "optimizer_llm" in cfg:
        opt_cfg = cfg["optimizer_llm"]
        model = opt_cfg.get("model", base_llm.get("model", "openai/gpt-oss-120b"))
        temperature = float(opt_cfg.get("temperature", 0.7))
        max_tokens = int(opt_cfg.get("max_tokens", 2048))
    else:
        model = base_llm.get("model", "openai/gpt-oss-20b")
        temperature = float(base_llm.get("temperature", 0.0))
        max_tokens = int(base_llm.get("max_tokens", 768))

    # If provider is gemini, set standard default model
    if provider == "gemini" and ("gpt-oss" in model or model == "groq" or "1.5" in model or "2.5" in model or "3.8" in model):
        model = "gemini-flash-lite-latest"

    key_envs = base_llm.get("key_envs")
    return LLMClient(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        requests_per_minute=base_llm.get("requests_per_minute", 28),
        timeout_s=base_llm.get("timeout_s", 60),
        cache_path=base_llm.get("cache_path"),
        key_envs=key_envs,
    )
