"""The agent: one model, one lead, one conversation. Handles the tool-calling loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .crm import CsvCRM
from .ollama_client import OllamaClient
from .prompts import agent_system_prompt
from .tools import TOOL_SCHEMAS, ToolExecutor


@dataclass
class AgentTurn:
    text: str
    tool_events: list = field(default_factory=list)  # [{"tool", "args", "result"}]
    ended: bool = False
    model_seconds: float = 0.0
    error: str = ""


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
        turn = AgentTurn(text="")
        spoken: list[str] = []
        for _ in range(self.max_tool_rounds):
            res = self.client.chat(self.model, self.messages, tools=TOOL_SCHEMAS,
                                   temperature=self.temperature, num_ctx=self.num_ctx)
            turn.model_seconds += res.seconds
            self.messages.append(res.raw_message)
            if res.content:
                spoken.append(res.content)
            if not res.tool_calls:
                break
            for call in res.tool_calls:
                result = self.tools.execute(call["name"], call["arguments"])
                turn.tool_events.append({"tool": call["name"], "args": call["arguments"], "result": result})
                self.messages.append({"role": "tool", "content": result, "tool_name": call["name"]})
            if self.tools.ended:
                break  # end_call was the last action; whatever was said before it is the goodbye
        else:
            turn.error = f"stopped after {self.max_tool_rounds} tool rounds without a spoken reply"
        turn.text = " ".join(spoken).strip()
        turn.ended = self.tools.ended
        return turn
