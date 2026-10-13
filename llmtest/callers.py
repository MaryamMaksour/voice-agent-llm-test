"""The 'customer' side of a test call: scripted lines, an LLM role-player, or you."""

from __future__ import annotations

from .ollama_client import OllamaClient
from .prompts import caller_system_prompt

END = "[END]"


class ScriptedCaller:
    """Plays fixed lines in order, whatever the agent says. Same input for every model = fair comparison."""

    kind = "scripted"

    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)

    def first_line(self) -> str:
        return self.lines.pop(0) if self.lines else "Hello?"

    def reply(self, agent_text: str) -> str | None:
        return self.lines.pop(0) if self.lines else None


class SimulatedCaller:
    """An LLM plays the customer from a persona. Realistic: reacts to what the agent actually asks."""

    kind = "simulated"

    def __init__(self, client: OllamaClient, model: str, persona: str, temperature: float, num_ctx: int) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.messages = [{"role": "system", "content": caller_system_prompt(persona)}]
        self.hung_up = False

    def first_line(self) -> str:
        line = "Hello?"
        self.messages.append({"role": "assistant", "content": line})
        return line

    def reply(self, agent_text: str) -> str | None:
        if self.hung_up:
            return None
        self.messages.append({"role": "user", "content": agent_text or "(silence)"})
        res = self.client.chat(self.model, self.messages, temperature=self.temperature, num_ctx=self.num_ctx)
        text = res.content.strip()
        self.messages.append({"role": "assistant", "content": text})
        if END in text:
            self.hung_up = True
            text = text.replace(END, "").strip()
        return text or None


class HumanCaller:
    """You type as the customer (used by chat.py)."""

    kind = "human"

    @staticmethod
    def _ask() -> str | None:
        try:
            return input("you (lead) > ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+Z / Ctrl+C = hang up
            print()
            return None

    def first_line(self) -> str:
        return self._ask() or "Hello?"

    def reply(self, agent_text: str) -> str | None:
        text = self._ask()
        return None if text is None or text.lower() in ("", "/end", "/quit") else text
