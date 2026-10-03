"""Model providers. Everything here uses only the standard library.

A provider is any object with ``name``/``model`` attributes and a
``complete(prompt, system=None) -> Completion`` method.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional

from .models import Completion

_RETRYABLE = {408, 429, 500, 502, 503, 504, 529}


class ProviderError(RuntimeError):
    pass


@dataclass
class Pricing:
    """USD per one million tokens. Defaults to free so cost is opt-in."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


def _post_json(url: str, headers: Dict[str, str], payload: dict, timeout: float, retries: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last = ProviderError(f"HTTP {exc.code}: {body[:300]}")
            if exc.code not in _RETRYABLE:
                raise last from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = ProviderError(f"network error: {exc}")
        if attempt < retries:
            time.sleep(min(2 ** attempt, 20) + random.random() * 0.25)
    assert last is not None
    raise last


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 120.0,
        retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def complete(self, prompt: str, system: Optional[str] = None) -> Completion:
        payload: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        body = _post_json(f"{self.base_url}/v1/messages", headers, payload, self.timeout, self.retries)
        text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        usage = body.get("usage", {})
        return Completion(
            text=text,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )


class OpenAICompatProvider:
    """Any server speaking the OpenAI chat-completions API (OpenAI, vLLM, Ollama, ...)."""

    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        timeout: float = 120.0,
        retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries

    def complete(self, prompt: str, system: Optional[str] = None) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        body = _post_json(f"{self.base_url}/chat/completions", headers, payload, self.timeout, self.retries)
        choices = body.get("choices") or []
        text = choices[0]["message"].get("content", "") if choices else ""
        usage = body.get("usage", {})
        return Completion(
            text=text or "",
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )


class MockProvider:
    """Offline provider for demos, CI and tests.

    Looks the prompt up in ``known`` (prompt -> reference answer) and returns it
    with probability ``accuracy``; otherwise returns a wrong answer. The
    decision is a deterministic hash of (seed, prompt), so runs are repeatable.
    """

    name = "mock"

    def __init__(
        self,
        known: Optional[Dict[str, str]] = None,
        accuracy: float = 1.0,
        seed: int = 0,
        latency_s: float = 0.0,
        model: str = "mock",
    ):
        self.known = known or {}
        self.accuracy = accuracy
        self.seed = seed
        self.latency_s = latency_s
        self.model = model

    def _is_correct(self, prompt: str) -> bool:
        digest = hashlib.sha256(f"{self.seed}:{prompt}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / 2 ** 64 < self.accuracy

    def complete(self, prompt: str, system: Optional[str] = None) -> Completion:
        if self.latency_s:
            time.sleep(self.latency_s)
        answer = self.known.get(prompt, prompt)
        text = answer if self._is_correct(prompt) else "I don't know."
        return Completion(
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
        )
