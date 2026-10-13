"""Minimal Ollama client (standard library only, no extra packages)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_IN_TEXT = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    tool_calls: list = field(default_factory=list)   # [{"name": str, "arguments": dict}]
    raw_message: dict = field(default_factory=dict)  # exactly what to append to the history
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0


def clean_text(text: str) -> str:
    """Remove <think> blocks and stray tags so they are never 'spoken'."""
    text = _THINK.sub("", text or "")
    text = text.replace("<think>", "").replace("</think>", "")
    return _TOOL_IN_TEXT.sub("", text).strip()


def _parse_arguments(args) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise OllamaError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e
        except urllib.error.URLError as e:
            raise OllamaError(f"cannot reach Ollama at {self.base_url}: {e.reason}") from e

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as e:
            raise OllamaError(f"cannot reach Ollama at {self.base_url}: {e}") from e

    def version(self) -> str:
        return self._get("/api/version").get("version", "?")

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._get("/api/tags").get("models", [])]

    def chat(self, model: str, messages: list[dict], tools: list | None = None,
             temperature: float = 0.2, num_ctx: int = 8192, think: bool = False) -> ChatResult:
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        if tools:
            body["tools"] = tools
        t0 = time.perf_counter()
        try:
            data = self._post("/api/chat", body)
        except OllamaError as e:
            # older Ollama, or a model without a thinking switch: retry without the flag
            if "think" not in str(e).lower():
                raise
            body.pop("think")
            data = self._post("/api/chat", body)
        seconds = time.perf_counter() - t0

        msg = data.get("message", {}) or {}
        content = msg.get("content", "") or ""
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("name"):
                calls.append({"name": fn["name"], "arguments": _parse_arguments(fn.get("arguments"))})
        if not calls:
            # some models write the tool call as text: <tool_call>{"name":..,"arguments":..}</tool_call>
            for m in _TOOL_IN_TEXT.finditer(content):
                try:
                    obj = json.loads(m.group(1))
                    calls.append({"name": obj.get("name", ""), "arguments": _parse_arguments(obj.get("arguments"))})
                except json.JSONDecodeError:
                    pass
        raw = {"role": "assistant", "content": content}
        if msg.get("tool_calls"):
            raw["tool_calls"] = msg["tool_calls"]
        return ChatResult(
            content=clean_text(content),
            tool_calls=calls,
            raw_message=raw,
            seconds=seconds,
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
        )
