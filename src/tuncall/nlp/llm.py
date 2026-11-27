"""One OpenAI-compatible client for Gemini and Groq, with every raw reply cached (D-32b, D-32c).

The cache key is the sha256 of the whole request (provider, model, messages, parameters), so a
reply is fetched once and scores can be recomputed offline. Rate limits (429) and server
overload (5xx) are waited out; a daily quota raises ``QuotaExhausted`` so long jobs can stop
and resume the next day from the cache.
"""

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import openai

log = logging.getLogger(__name__)

PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}
MAX_WAIT = 300.0
GIVE_UP_AFTER = 3600.0


class QuotaExhausted(RuntimeError):
    """The provider's daily free-tier quota is used up; rerun later to resume."""


def _daily(err: openai.RateLimitError) -> bool:
    text = str(err).lower()
    return "per day" in text or "perday" in text or "daily" in text


class LLM:
    def __init__(
        self,
        provider: str,
        model: str,
        cache: Path,
        *,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider, self.model, self.cache = provider, model, cache
        if client is None:
            base_url, key = PROVIDERS[provider]
            client = openai.OpenAI(base_url=base_url, api_key=os.environ[key], max_retries=0)
        self._client, self._sleep = client, sleep

    def key(self, messages: list[dict[str, str]], params: dict[str, Any]) -> str:
        request = {"provider": self.provider, "model": self.model, "messages": messages, **params}
        return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

    def chat(self, messages: list[dict[str, str]], **params: Any) -> str:
        """The reply text, from the cache or from the API (then cached)."""
        path = self.cache / f"{self.key(messages, params)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["reply"]
        start, wait = time.monotonic(), 10.0
        while True:
            try:
                t0 = time.monotonic()
                resp = self._client.chat.completions.create(
                    model=self.model, messages=messages, **params
                )
                seconds = time.monotonic() - t0
                break
            except openai.RateLimitError as err:
                if _daily(err):
                    raise QuotaExhausted(str(err)) from err
                retry = err
            except (openai.APIConnectionError, openai.InternalServerError) as err:
                retry = err
            if time.monotonic() - start > GIVE_UP_AFTER:
                raise retry
            log.warning("%s %s: %s; retrying in %.0f s", self.provider, self.model, retry, wait)
            self._sleep(wait)
            wait = min(wait * 2, MAX_WAIT)
        reply = resp.choices[0].message.content or ""
        record = {"model": self.model, "messages": messages, "params": params}
        record |= {"reply": reply, "seconds": seconds}
        self.cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return reply


def parse_json(reply: str) -> Any:
    """The JSON value in a reply, tolerating a ```json fence; ``None`` if it is not valid JSON."""
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
