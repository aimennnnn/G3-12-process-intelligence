"""One small client for every free LLM option.

Groq, Gemini and Ollama all speak the OpenAI chat-completions format, so a
single HTTP call covers them. Adds: rate limiting (free tiers cap requests per
minute), retries on 429/5xx, and a disk cache so re-running an experiment
never spends quota twice. Every call reports tokens and latency for the
cost/latency metrics.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from src.config import load_dotenv

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "GEMINI_API_KEY"),
    "ollama": ("http://localhost:11434/v1/chat/completions", None),
}


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


class LLMClient:
    def __init__(self, provider: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 16, requests_per_minute: int = 28,
                 timeout_s: int = 60, cache_path: str | None = None):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Use one of {list(PROVIDERS)} or mock.")
        load_dotenv()
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens, self.timeout_s = temperature, max_tokens, timeout_s
        self.url, key_name = PROVIDERS[provider]
        self.api_key = os.environ.get(key_name) if key_name else None
        if key_name and not self.api_key:
            raise RuntimeError(f"{key_name} is not set. Put it in a .env file (see .env.example).")
        self.min_interval = 60.0 / max(requests_per_minute, 1)
        self._last_call = 0.0
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = self._load_cache()

    # ---------- cache ----------
    def _key(self, messages: list[dict]) -> str:
        blob = json.dumps([self.provider, self.model, self.temperature, messages], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _load_cache(self) -> dict:
        cache = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    cache[row["key"]] = row
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
        key = self._key(messages)
        if key in self._cache:
            c = self._cache[key]
            return LLMResponse(c["text"], c["prompt_tokens"], c["completion_tokens"],
                               c["latency_s"], cached=True)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}

        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            t0 = time.time()
            try:
                r = requests.post(self.url, headers=headers, json=body, timeout=self.timeout_s)
            except requests.RequestException as e:
                self._backoff(attempt, f"network error: {e}")
                continue
            if r.status_code == 429 or r.status_code >= 500:
                retry_after = float(r.headers.get("retry-after", 0) or 0)
                self._backoff(attempt, f"HTTP {r.status_code}", retry_after)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"LLM call failed ({r.status_code}): {r.text[:300]}")
            data = r.json()
            usage = data.get("usage") or {}
            resp = LLMResponse(
                text=(data["choices"][0]["message"].get("content") or "").strip(),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                latency_s=round(time.time() - t0, 3),
            )
            self._save(key, resp)
            return resp
        raise RuntimeError("LLM call kept failing after 6 attempts (rate limit or outage).")

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

    def chat(self, messages: list[dict]) -> LLMResponse:
        prompt = messages[-1]["content"]
        rng = random.Random(hashlib.sha256(prompt.encode()).hexdigest())
        if "ANOMALOUS" in prompt:
            text = rng.choice(["NORMAL", "ANOMALOUS"])
        else:
            n = prompt.count("\n[")  # numbered options look like "[3] ..."
            text = str(rng.randint(1, max(n, 1)))
        return LLMResponse(text, prompt_tokens=len(prompt) // 4, completion_tokens=1)


def make_llm(cfg: dict, provider: str | None = None):
    c = dict(cfg["llm"])
    provider = provider or c["provider"]
    if provider == "mock":
        return MockLLM()
    return LLMClient(provider, c["model"], c["temperature"], c["max_tokens"],
                     c["requests_per_minute"], c["timeout_s"], c["cache_path"])
