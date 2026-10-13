"""Tools the agent can call, and the code that runs them against the CSV CRM.

Each call is bound to ONE lead: the model can never change another lead's row."""

from __future__ import annotations

from datetime import datetime

from .crm import INTEREST_LEVELS, PROPERTY_TYPES, CRMError, CsvCRM

STATUS_CHOICES = ["interested", "qualified", "not_interested", "wrong_number"]

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "update_lead",
            "description": "Save facts about the lead to the CRM as soon as the lead says them. "
                           "Pass only the fields you just learned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": STATUS_CHOICES,
                               "description": "Lead status after what the lead said."},
                    "budget_aed": {"type": "integer",
                                   "description": "Budget in whole UAE dirhams, e.g. 2000000. "
                                                  "For a range, use the upper limit."},
                    "preferred_area": {"type": "string",
                                       "description": "Area the lead wants, in their words, e.g. 'Dubai Marina'."},
                    "property_type": {"type": "string", "enum": PROPERTY_TYPES},
                    "timeline": {"type": "string",
                                 "description": "When they plan to buy, in their words, e.g. 'within 3 months'."},
                    "interest_level": {"type": "string", "enum": INTEREST_LEVELS,
                                       "description": "hot = buying within 3 months, warm = 3 to 12 months, "
                                                      "cold = more than 12 months or unsure."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_callback",
            "description": "Book a callback when the lead (or the person answering) asks to be called later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "datetime": {"type": "string",
                                 "description": "Agreed day and time as YYYY-MM-DDTHH:MM (24-hour, Dubai time)."},
                },
                "required": ["datetime"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_do_not_call",
            "description": "Use when the lead asks not to be called again.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "Hang up. Always the last action of every call, after saying goodbye.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string",
                                "description": "One sentence for the sales team: what happened on the call."},
                },
                "required": ["summary"],
            },
        },
    },
]


class ToolExecutor:
    def __init__(self, crm: CsvCRM, lead_id: str, now: datetime) -> None:
        self.crm = crm
        self.lead_id = lead_id
        self.now = now
        self.ended = False
        self.actions: list[dict] = []  # audit trail of every successful CRM write

    def execute(self, name: str, args: dict) -> str:
        """Run one tool. Always returns a short text result for the model (errors included)."""
        try:
            if name == "update_lead":
                row_before = self.crm.get(self.lead_id)
                self.crm.update(self.lead_id, args)
                changed = {k: v for k, v in args.items() if v not in (None, "") and str(row_before.get(k)) != str(v)}
                self._log(name, args)
                return f"Saved: {', '.join(changed) or 'no change'}."
            if name == "schedule_callback":
                raw = str(args.get("datetime", "")).strip()
                try:
                    when = datetime.fromisoformat(raw[:16])
                except ValueError:
                    return "Not saved: datetime must look like 2026-10-15T16:00. Ask the lead for the day and time."
                if when <= self.now:
                    return f"Not saved: {raw} is in the past (now is {self.now:%Y-%m-%d %H:%M}). Confirm the day and time."
                self.crm.schedule_callback(self.lead_id, when)
                self._log(name, {"datetime": when.strftime("%Y-%m-%dT%H:%M")})
                return f"Callback booked for {when:%A %d %B at %H:%M}."
            if name == "mark_do_not_call":
                self.crm.mark_do_not_call(self.lead_id)
                self._log(name, {})
                return "Marked as do-not-call."
            if name == "end_call":
                summary = str(args.get("summary", "")).strip() or "(no summary)"
                self.crm.add_note(self.lead_id, summary)
                self.ended = True
                self._log(name, {"summary": summary})
                return "Call ended."
            return f"Unknown tool '{name}'. Available: update_lead, schedule_callback, mark_do_not_call, end_call."
        except CRMError as e:
            return f"Not saved: {e}"

    def _log(self, name: str, args: dict) -> None:
        self.actions.append({"tool": name, "args": args})
