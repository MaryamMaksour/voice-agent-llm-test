"""Post-call check: after the call, read the conversation once more and fill in details that were said
but not saved. Only empty fields are filled; nothing the agent saved is overwritten."""

from __future__ import annotations

import json

from .crm import PROPERTY_TYPES, CRMError, CsvCRM, validate_changes
from .ollama_client import OllamaClient, OllamaError
from .rules import DETAIL_FIELDS, apply_rules, is_vague_area

SKIP_STATUSES = {"not_interested", "wrong_number", "do_not_call", "callback"}

SCHEMA = {
    "type": "object",
    "properties": {
        "still_interested": {"type": ["boolean", "null"]},
        "budget_aed": {"type": ["integer", "null"]},
        "preferred_area": {"type": ["string", "null"]},
        "property_type": {"type": ["string", "null"]},
        "timeline": {"type": ["string", "null"]},
    },
    "required": ["still_interested", "budget_aed", "preferred_area", "property_type", "timeline"],
}

PROMPT = f"""You read the transcript of a phone call between a real estate agent and a lead.
Extract ONLY what the LEAD clearly said. Use null for anything the lead did not say. Never guess.
- still_interested: true if the lead said they are still interested, false if not, null if unclear.
- budget_aed: whole number of dirhams ("two million" = 2000000). For a range, the upper limit.
- preferred_area: the area the lead named. null if they said anywhere, any area or that they do not mind.
- property_type: one of {", ".join(PROPERTY_TYPES)}.
- timeline: when they plan to buy, in their own words.
Answer with the JSON object only."""


def _conversation(events: list[dict]) -> str:
    lines = [f"{'LEAD' if e['type'] == 'lead' else 'AGENT'}: {e['text']}"
             for e in events if e["type"] in ("lead", "agent") and e["text"] != "(no spoken reply)"]
    return "\n".join(lines)


def post_call_check(client: OllamaClient, model: str, crm: CsvCRM, lead_id: str, events: list[dict],
                    *, num_ctx: int) -> tuple[dict, str]:
    """Returns (fields changed, one line for the log)."""
    row = crm.get(lead_id)
    if row.get("status") in SKIP_STATUSES or row.get("do_not_call") == "true":
        return {}, f"post-call check: skipped (status {row.get('status')})"
    missing = [f for f in DETAIL_FIELDS if not (row.get(f) or "").strip()]
    if not missing:
        ruled = apply_rules(crm, lead_id)
        return ruled, "post-call check: all details already saved" + (f"; rules set {ruled}" if ruled else "")

    system = PROMPT + ("\n/no_think" if "qwen3" in model.lower() else "")
    try:
        res = client.chat(model, [{"role": "system", "content": system},
                                  {"role": "user", "content": _conversation(events)}],
                          temperature=0, num_ctx=num_ctx, fmt=SCHEMA)
        found = json.loads(res.content)
    except (OllamaError, json.JSONDecodeError, TypeError) as e:
        return {}, f"post-call check: no usable answer ({e})"
    if not isinstance(found, dict):
        return {}, "post-call check: no usable answer"

    fills = {}
    for field in missing:
        value = found.get(field)
        if value in (None, "") or (field == "preferred_area" and is_vague_area(str(value))):
            continue
        try:
            fills.update(validate_changes({field: value}))
        except CRMError:
            continue  # e.g. a property type outside the allowed list: better empty than wrong
    if found.get("still_interested") is True and (row.get("status") or "new") in ("", "new", "contacted"):
        fills["status"] = "interested"
    if fills:
        crm.update(lead_id, fills)
    changed = fills | apply_rules(crm, lead_id)
    note = ", ".join(f"{k}={v}" for k, v in changed.items()) or "nothing to add"
    return changed, f"post-call check: {note}"
