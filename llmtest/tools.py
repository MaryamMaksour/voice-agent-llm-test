"""Tools the agent can call, and the code that runs them against the CSV CRM.

Each call is bound to ONE lead: the model can never change another lead's row."""

from __future__ import annotations

from datetime import datetime

from .crm import INTEREST_LEVELS, PROPERTY_TYPES, CRMError, CsvCRM
from .rules import apply_rules, is_vague_area, resolve_callback

# "qualified" is not offered: the code sets it once all four details are known (rules.derive_status)
STATUS_CHOICES = ["interested", "not_interested", "wrong_number"]

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "update_lead",
            "description": "Save facts about the lead to the CRM as soon as the lead says them. "
                           "Pass EVERY field the lead just gave, all in this one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": STATUS_CHOICES,
                               "description": "Only for: not interested, or wrong number. "
                                              "Interested and qualified are set automatically."},
                    "budget_aed": {"type": "integer",
                                   "description": "Budget in whole UAE dirhams, e.g. 2000000. "
                                                  "For a range, use the upper limit."},
                    "preferred_area": {"type": "string",
                                       "description": "Area the lead wants, e.g. 'Dubai Marina'. "
                                                      "Leave out if the lead does not mind the area."},
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
                    "day": {"type": "string",
                            "description": "The day the lead said: 'today', 'tomorrow', a weekday such as "
                                           "'monday' (means the next Monday), or a date YYYY-MM-DD."},
                    "time": {"type": "string",
                             "description": "The time as HH:MM, 24-hour, e.g. '16:00'. Morning = 10:00, "
                                            "afternoon = 15:00, evening = 18:00."},
                },
                "required": ["day", "time"],
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
        self.last_rules: dict = {}     # what the business rules changed after the last tool call

    def execute(self, name: str, args: dict) -> str:
        """Run one tool. Always returns a short text result for the model (errors included)."""
        self.last_rules = {}
        try:
            if name == "update_lead":
                args = dict(args)
                note = ""
                if is_vague_area(str(args.get("preferred_area") or "")):
                    args.pop("preferred_area")
                    note = " Area not saved: the lead has no preferred area."
                row_before = self.crm.get(self.lead_id)
                self.crm.update(self.lead_id, args)
                changed = {k: v for k, v in args.items() if v not in (None, "") and str(row_before.get(k)) != str(v)}
                self._log(name, args)
                self.last_rules = apply_rules(self.crm, self.lead_id)
                if self.last_rules:
                    self._log("rules", self.last_rules)
                return f"Saved: {', '.join(changed) or 'no change'}.{note}"
            if name == "schedule_callback":
                try:
                    if args.get("day"):
                        when = resolve_callback(str(args["day"]), str(args.get("time", "")), self.now)
                    else:  # older form: one ISO datetime
                        when = datetime.fromisoformat(str(args.get("datetime", "")).strip()[:16])
                        if when <= self.now:
                            raise ValueError(f"{when:%Y-%m-%d %H:%M} is in the past (now is {self.now:%Y-%m-%d %H:%M})")
                except ValueError as e:
                    return f"Not saved: {e}. Ask the lead for the day and time."
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
