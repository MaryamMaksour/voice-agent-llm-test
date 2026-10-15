"""Runs one call between an agent and a caller, logging every line and tool call."""

from __future__ import annotations

from dataclasses import dataclass

from .agent import Agent
from .transcript import Transcript


@dataclass
class CallResult:
    ended_by: str          # "end_call" | "caller_hung_up" | "max_turns" | "error"
    agent_turns: int
    model_seconds: float
    error: str = ""


def run_call(agent: Agent, caller, transcript: Transcript, max_turns: int) -> CallResult:
    lead_line = caller.first_line()
    total_seconds = 0.0
    turns = 0
    final_chance_used = False
    while True:
        transcript.lead(lead_line)
        try:
            turn = agent.respond(lead_line)
        except Exception as e:  # a model/server error must not kill the whole test run
            transcript.note(f"ERROR: {e}")
            return CallResult("error", turns, total_seconds, str(e))
        turns += 1
        total_seconds += turn.model_seconds
        for ev in turn.tool_events:
            transcript.tool(ev["tool"], ev["args"], ev["result"])
            if ev.get("rules"):
                transcript.note("rules set: " + ", ".join(f"{k}={v}" for k, v in ev["rules"].items()))
        for sentence in turn.guarded:
            transcript.note(f"guard: removed a price the lead never gave: {sentence!r}")
        transcript.agent(turn.text or "(no spoken reply)", turn.model_seconds)
        if turn.error:
            transcript.note(turn.error)
        if turn.ended:
            return CallResult("end_call", turns, total_seconds)
        if turns >= max_turns:
            transcript.note(f"stopped: agent did not end the call within {max_turns} replies")
            return CallResult("max_turns", turns, total_seconds)

        next_line = caller.reply(turn.text)
        if next_line is None:
            if final_chance_used:
                return CallResult("caller_hung_up", turns, total_seconds)
            # the lead said goodbye: give the agent one last turn to save data and call end_call
            final_chance_used = True
            next_line = "(the lead has said goodbye and is about to hang up)"
        lead_line = next_line
