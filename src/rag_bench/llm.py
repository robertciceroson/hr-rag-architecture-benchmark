"""
Minimal LLM client for the LLM-assisted experiments (HyDE, LLM scope gate).

Uses Groq. The model is set by GROQ_MODEL in .env (default: openai/gpt-oss-120b).
Groq retired llama-3.3-70b-versatile (the production bot's original model), so
the model is a setting, not a code constant: a future retirement is a one-line
.env change. List available models with `python scripts/list_models.py`.
Every completion is cached to disk keyed by (model, prompt), so:
  - re-running an experiment costs nothing and is fully reproducible
  - the cache file can be committed as evidence of exactly what the LLM said
"""
import hashlib
import json
import os
from typing import Optional

DEFAULT_MODEL = "openai/gpt-oss-120b"

# Reasoning models spend completion tokens "thinking" before they answer. A tight
# max_tokens (the scope gate asks for one word) would leave the visible answer
# EMPTY, which would silently read as "in scope". Give them room, keep effort low.
REASONING_MODEL_PREFIXES = ("openai/gpt-oss", "qwen/qwen3")
REASONING_MIN_TOKENS = 1024


class CachedLLM:
    def __init__(self, cache_path: str, model: Optional[str] = None, client=None):
        self.model = model or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
        self.reasoning = self.model.startswith(REASONING_MODEL_PREFIXES)
        self.cache_path = cache_path
        self.calls = 0          # real API calls this run
        self.cache_hits = 0
        self._client = client
        self._cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self._cache = json.load(f)

    @property
    def client(self):
        if self._client is None:
            from groq import Groq  # imported lazily so non-LLM runs don't need it
            if not os.environ.get("GROQ_API_KEY"):
                raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
            self._client = Groq()
        return self._client

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 300) -> str:
        key = hashlib.sha256(f"{self.model}\n{system}\n{prompt}".encode()).hexdigest()
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        kwargs = {"model": self.model, "messages": messages, "temperature": 0,
                  "max_tokens": max_tokens}
        if self.reasoning:
            kwargs["max_tokens"] = max(max_tokens, REASONING_MIN_TOKENS)
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        resp = self.client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            # Never cache or return an empty answer: callers would misread it.
            raise RuntimeError(
                f"{self.model} returned an empty answer (finish_reason="
                f"{resp.choices[0].finish_reason}). Nothing was cached; rerun, or "
                f"set a different GROQ_MODEL in .env.")
        self.calls += 1
        self._cache[key] = text
        self._save()
        return text

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=1)
