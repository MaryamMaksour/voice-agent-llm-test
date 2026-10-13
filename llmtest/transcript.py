"""Conversation logs: one JSONL file (machine-readable) and one TXT file (easy to read) per call."""

from __future__ import annotations

import json
import os
from datetime import datetime


class Transcript:
    def __init__(self, folder: str, name: str, header: dict) -> None:
        os.makedirs(folder, exist_ok=True)
        self.jsonl_path = os.path.join(folder, f"{name}.jsonl")
        self.txt_path = os.path.join(folder, f"{name}.txt")
        self.events: list[dict] = []
        self._jsonl = open(self.jsonl_path, "w", encoding="utf-8")
        self._txt = open(self.txt_path, "w", encoding="utf-8")
        self._emit({"type": "header", **header})
        self._txt.write("\n".join(f"{k}: {v}" for k, v in header.items()) + "\n" + "-" * 70 + "\n")

    def _emit(self, event: dict) -> None:
        event = {"time": datetime.now().isoformat(timespec="seconds"), **event}
        self.events.append(event)
        self._jsonl.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._jsonl.flush()

    def lead(self, text: str) -> None:
        self._emit({"type": "lead", "text": text})
        self._txt.write(f"LEAD : {text}\n")

    def agent(self, text: str, seconds: float) -> None:
        self._emit({"type": "agent", "text": text, "model_seconds": round(seconds, 2)})
        self._txt.write(f"AGENT: {text}   ({seconds:.1f}s)\n")

    def tool(self, name: str, args: dict, result: str) -> None:
        self._emit({"type": "tool", "tool": name, "args": args, "result": result})
        self._txt.write(f"   [tool] {name}({json.dumps(args, ensure_ascii=False)}) -> {result}\n")

    def note(self, text: str) -> None:
        self._emit({"type": "note", "text": text})
        self._txt.write(f"   [note] {text}\n")

    def close(self, footer: dict) -> None:
        self._emit({"type": "footer", **footer})
        self._txt.write("-" * 70 + "\n" + "\n".join(f"{k}: {v}" for k, v in footer.items()) + "\n")
        self._jsonl.close()
        self._txt.close()
