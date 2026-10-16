"""Conversation logs: one JSONL file (machine-readable) and one TXT file (easy to read) per call."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime


class Transcript:
    def __init__(self, folder: str, name: str, header: dict) -> None:
        os.makedirs(folder, exist_ok=True)
        self.jsonl_path = os.path.join(folder, f"{name}.jsonl")
        self.txt_path = os.path.join(folder, f"{name}.txt")
        self.events: list[dict] = []
        self._started = time.perf_counter()
        self._jsonl = open(self.jsonl_path, "w", encoding="utf-8")
        self._txt = open(self.txt_path, "w", encoding="utf-8")
        self._emit({"type": "header", **header})
        self._txt.write("\n".join(f"{k}: {v}" for k, v in header.items()) + "\n" + "-" * 70 + "\n")

    def _emit(self, event: dict) -> None:
        # local time WITH its UTC offset, so the time is unambiguous on any machine
        event = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), **event}
        self.events.append(event)
        self._jsonl.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._jsonl.flush()

    def lead(self, text: str) -> None:
        self._emit({"type": "lead", "text": text})
        self._txt.write(f"LEAD : {text}\n")

    def agent(self, text: str, seconds: float, *, reply_seconds: float | None = None,
              rounds: list | None = None, thinking: list | None = None) -> None:
        """seconds = model time; reply_seconds = how long the lead waited (model calls + tools)."""
        event = {"type": "agent", "text": text, "model_seconds": round(seconds, 2)}
        if reply_seconds is not None:
            event["reply_seconds"] = round(reply_seconds, 2)
        if rounds:
            event["rounds"] = rounds
        if thinking:
            event["thinking"] = thinking
        self._emit(event)
        waited = reply_seconds if reply_seconds is not None else seconds
        extra = f", {len(rounds)} model calls" if rounds and len(rounds) > 1 else ""
        self._txt.write(f"AGENT: {text}   ({waited:.1f}s{extra})\n")
        for t in thinking or []:
            self._txt.write(f"   [thinking] {t}\n")

    def tool(self, name: str, args: dict, result: str, *, seconds: float | None = None,
             ok: bool | None = None) -> None:
        event = {"type": "tool", "tool": name, "args": args, "result": result}
        if seconds is not None:
            event["seconds"] = seconds
        if ok is not None:
            event["ok"] = ok
        self._emit(event)
        self._txt.write(f"   [tool] {name}({json.dumps(args, ensure_ascii=False)}) -> {result}\n")

    def note(self, text: str) -> None:
        self._emit({"type": "note", "text": text})
        self._txt.write(f"   [note] {text}\n")

    def close(self, footer: dict) -> None:
        footer = {"duration_seconds": round(time.perf_counter() - self._started, 1), **footer}
        self._emit({"type": "footer", **footer})
        self._txt.write("-" * 70 + "\n" + "\n".join(f"{k}: {v}" for k, v in footer.items()) + "\n")
        self._jsonl.close()
        self._txt.close()
