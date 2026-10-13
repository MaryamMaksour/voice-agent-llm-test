"""Scores one call: did the CRM row end up correct, and did the agent behave like a good phone agent?"""

from __future__ import annotations

import re

FIELD_KEYS = ["status", "budget_aed", "preferred_area", "property_type", "timeline",
              "interest_level", "callback_at", "do_not_call"]
_PRICE_QUESTION = re.compile(r"how much|price|cost|expensive|cheap", re.I)
_NUMBER = re.compile(r"\d|million|thousand|hundred", re.I)


def _norm(row: dict, key: str):
    value = (row.get(key) or "").strip()
    if value == "":
        return None
    if key == "budget_aed":
        try:
            return int(float(value))
        except ValueError:
            return value
    if key == "do_not_call":
        return value.lower() == "true"
    return value


def field_checks(expected: dict, final_row: dict) -> list[dict]:
    rows = []
    for key in FIELD_KEYS:
        want = expected.get(key)
        got = _norm(final_row, key)
        if key == "do_not_call":
            want, got = bool(want), bool(got)
            ok = want == got
        elif key == "preferred_area":
            # free text: any of the accepted spellings must appear; null means "must not be saved"
            ok = got is None if not want else got is not None and any(w.lower() in got.lower() for w in want)
        elif key == "timeline":
            ok = (got is None) if not want else got is not None  # wording varies, presence is what matters
        elif key == "callback_at":
            ok = (got is None) if not want else got is not None and got[:16] == want[:16]
        else:
            ok = (got is None) if want is None else str(got).lower() == str(want).lower()
        rows.append({"field": key, "expected": want, "got": got, "ok": ok})
    return rows


def conduct_checks(events: list[dict], ended_by: str) -> dict:
    agent_lines = [e["text"] for e in events if e["type"] == "agent" and e["text"] != "(no spoken reply)"]
    n = max(len(agent_lines), 1)
    short = sum(1 for t in agent_lines if len(t.split()) <= 40) / n
    one_q = sum(1 for t in agent_lines if t.count("?") <= 1) / n
    # invented price = the lead asked about price and the agent answered with a number
    invented, last_lead = [], ""
    for e in events:
        if e["type"] == "lead":
            last_lead = e["text"]
        elif e["type"] == "agent" and _PRICE_QUESTION.search(last_lead) and _NUMBER.search(e["text"]):
            invented.append(e["text"])
    tool_errors = [e for e in events if e["type"] == "tool" and
                   (e["result"].startswith("Not saved") or e["result"].startswith("Unknown tool"))]
    return {
        "ended_with_end_call": ended_by == "end_call",
        "short_replies_share": round(short, 2),
        "one_question_share": round(one_q, 2),
        "invented_money_lines": invented,
        "tool_errors": len(tool_errors),
    }


def evaluate(scenario: dict, final_row: dict, events: list[dict], ended_by: str) -> dict:
    fields = field_checks(scenario["expected"], final_row)
    conduct = conduct_checks(events, ended_by)
    passed = all(f["ok"] for f in fields) and conduct["ended_with_end_call"] and not conduct["invented_money_lines"]
    return {"fields": fields, "conduct": conduct, "passed": passed}
