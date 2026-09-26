"""
Minimal LLM client for the LLM-assisted experiments (HyDE, LLM scope gate).

Uses Groq (Llama 3.3 70B) to match the production HR-Policy-QA-Bot.
Every completion is cached to disk keyed by (model, prompt), so:
  - re-running an experiment costs nothing and is fully reproducible
  - the cache file can be committed as evidence of exactly what the LLM said
"""
import hashlib
import json
import os
from typing import Optional

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class CachedLLM:
    def __init__(self, cache_path: str, model: str = DEFAULT_MODEL, client=None):
        self.model = model
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
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=max_tokens)
        text = resp.choices[0].message.content.strip()
        self.calls += 1
        self._cache[key] = text
        self._save()
        return text

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=1)
