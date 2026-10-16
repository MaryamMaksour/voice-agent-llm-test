"""What a call achieved, in one word: saved in every call log's footer so the dashboard can group calls.

The result comes from the lead's CRM row after the call; "ended" says whether the agent closed it properly.
"""

from __future__ import annotations

# lead status after the call -> call classification
CLASSIFICATION = {
    "qualified": "Qualified",
    "interested": "Interested, details missing",
    "callback": "Callback booked",
    "not_interested": "Not interested",
    "do_not_call": "Do not call",
    "wrong_number": "Wrong number",
}
NO_RESULT = "No result saved"

ENDINGS = {
    "end_call": "Agent closed the call",
    "caller_hung_up": "Lead hung up first",
    "max_turns": "Stopped: too many turns",
    "error": "Stopped: error",
}


def classify_call(row: dict, ended_by: str) -> dict:
    status = (row or {}).get("status") or ""
    return {
        "result": status or None,
        "classification": CLASSIFICATION.get(status, NO_RESULT),
        "agent_closed": ended_by == "end_call",
        "ending": ENDINGS.get(ended_by, ended_by),
    }
