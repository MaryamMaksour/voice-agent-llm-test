"""The agent: one model, one lead, one conversation. Handles the tool-calling loop."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .crm import CsvCRM
from .ollama_client import OllamaClient
from .prompts import agent_system_prompt
from .rules import guard_money, guard_tool_talk, money_amounts
from .tools import TOOL_SCHEMAS, ToolExecutor


@dataclass
class AgentTurn:
    text: str
    tool_events: list = field(default_factory=list)  # [{"tool", "args", "result"}]
    ended: bool = False
    model_seconds: float = 0.0
    error: str = ""
    guarded: list = field(default_factory=list)  # sentences removed by the guards (prices, tool talk)


class Agent:
    def __init__(self, client: OllamaClient, model: str, crm: CsvCRM, lead_id: str, now: datetime,
                 *, company: str, agent_name: str, currency: str, temperature: float, num_ctx: int,
                 max_tool_rounds: int) -> None:
        self.client = client
        self.model = model
        self.tools = ToolExecutor(crm, lead_id, now)
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_tool_rounds = max_tool_rounds
        lead = crm.get(lead_id)
        self.system_prompt = agent_system_prompt(
            lead, company=company, agent_name=agent_name, now=now, currency=currency, model=model)
        self.messages: list[dict] = [{"role": "system", "content": self.system_prompt}]

    def respond(self, lead_text: str) -> AgentTurn:
        """Give the agent what the lead just said; return what the agent says back."""
        self.messages.append({"role": "user", "content": lead_text})
        self.tools.last_lead_text = lead_text
        turn = AgentTurn(text="")
        spoken: list[str] = []
        for _ in range(self.max_tool_rounds):
            res = self.client.chat(self.model, self.messages, tools=TOOL_SCHEMAS,
                                   temperature=self.temperature, num_ctx=self.num_ctx)
            turn.model_seconds += res.seconds
            content = res.content
            if content:
                content, talk = guard_tool_talk(content)
                content, removed = guard_money(content, self._lead_amounts())
                removed = talk + removed
                if removed:
                    # the history must match what the lead heard, or the model repeats the price later;
                    # keep any tool call the model wrote as text
                    blocks = re.findall(r"<tool_call>.*?</tool_call>", res.raw_message["content"], re.DOTALL)
                    res.raw_message["content"] = "\n".join([content, *blocks])
                    turn.guarded.extend(removed)
            self.messages.append(res.raw_message)
            if content:
                spoken.append(content)
            if not res.tool_calls:
                break
            for call in res.tool_calls:
                result = self.tools.execute(call["name"], call["arguments"])
                turn.tool_events.append({"tool": call["name"], "args": call["arguments"], "result": result,
                                         "rules": self.tools.last_rules})
                self.messages.append({"role": "tool", "content": result, "tool_name": call["name"]})
            if self.tools.ended:
                break  # end_call was the last action; whatever was said before it is the goodbye
        else:
            turn.error = f"stopped after {self.max_tool_rounds} tool rounds without a spoken reply"
        turn.text = " ".join(spoken).strip()
        turn.ended = self.tools.ended
        return turn

    def _lead_amounts(self) -> set[int]:
        """Amounts of money the lead said (the agent may repeat those, e.g. to confirm the budget)."""
        amounts: set[int] = set()
        for m in self.messages:
            if m["role"] == "user":
                amounts |= money_amounts(m["content"])
        return amounts
