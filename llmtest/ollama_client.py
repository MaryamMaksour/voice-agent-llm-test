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
    # where the time went, from Ollama's own timings (0 when the server doesn't report them)
    load_seconds: float = 0.0       # loading the model into memory (a "cold start")
    prompt_seconds: float = 0.0     # reading the prompt: grows with the length of the conversation
    generate_seconds: float = 0.0   # writing the reply: grows with the number of output tokens
    thinking: str = ""              # the model's thinking text, if it produced any (it is never spoken)
    tool_calls_as_text: bool = False  # the model wrote its tool call as text instead of a real tool call


def thinking_text(text: str) -> str:
    """The text inside <think>...</think>, which clean_text removes before anything is spoken."""
    return "\n".join(m.strip() for m in re.findall(r"<think>(.*?)</think>", text or "", re.DOTALL) if m.strip())


def _seconds(ns) -> float:
    return round((ns or 0) / 1e9, 3)


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
             temperature: float = 0.2, num_ctx: int = 8192, think: bool = False,
             fmt: dict | None = None) -> ChatResult:
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        if tools:
            body["tools"] = tools
        if fmt:
            body["format"] = fmt  # JSON schema: Ollama forces the reply into this shape
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
        as_text = False
        if not calls:
            # some models write the tool call as text: <tool_call>{"name":..,"arguments":..}</tool_call>
            for m in _TOOL_IN_TEXT.finditer(content):
                as_text = True
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
            load_seconds=_seconds(data.get("load_duration")),
            prompt_seconds=_seconds(data.get("prompt_eval_duration")),
            generate_seconds=_seconds(data.get("eval_duration")),
            thinking=(msg.get("thinking") or "").strip() or thinking_text(content),
            tool_calls_as_text=as_text and bool(calls),
        )
