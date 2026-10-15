"""Business rules applied in code, so the model does not have to get them right on its own.

- Status: worked out from the details collected (qualified only when all four are known).
- Interest level: worked out from the timeline (hot / warm / cold).
- Callback date: the model says "monday" + "10:00"; the code turns that into 2026-10-19T10:00.
- Area: "anywhere" / "I don't mind" is not an area, so it is never saved.
- Money guard: a spoken sentence with an amount of money the lead did not say is removed.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

# --- status -------------------------------------------------------------------------------------

DETAIL_FIELDS = ["budget_aed", "preferred_area", "property_type", "timeline"]
OPEN_STATUSES = {"", "new", "contacted", "interested", "qualified"}  # final outcomes are never overwritten


def derive_status(row: dict) -> str | None:
    """qualified = all four details known; interested = some known. None = leave the status as it is."""
    status = (row.get("status") or "").strip()
    if status not in OPEN_STATUSES:
        return None
    known = [bool((row.get(f) or "").strip()) for f in DETAIL_FIELDS]
    if all(known):
        return "qualified"
    if any(known) or status == "qualified":
        return "interested"
    return None


# --- interest level from the timeline ----------------------------------------------------------

_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "eighteen": 18, "twenty": 20, "couple": 2, "few": 3,
    "several": 4,
}
_UNIT_MONTHS = {"day": 1 / 30, "week": 12 / 52, "month": 1, "year": 12}
_DURATION = re.compile(
    r"\b(\d+(?:\.\d+)?|" + "|".join(_WORD_NUMBERS) + r")(\s+and\s+a\s+half)?\s*(?:of\s+)?(day|week|month|year)s?\b",
    re.I)
_SOON = re.compile(r"\b(asap|as soon as possible|immediately|right away|right now|this month|this week|urgent)", re.I)
_UNSURE = re.compile(r"\b(not sure|unsure|don'?t know|no idea|just browsing|no plans?|no rush|someday)\b", re.I)


def timeline_months(text: str) -> float | None:
    """The longest duration mentioned, in months ('within 3 to 6 months' -> 6). None if there is none."""
    durations = []
    for number, half, unit in _DURATION.findall(text or ""):
        n = float(number) if number[0].isdigit() else _WORD_NUMBERS[number.lower()]
        if half:
            n += 0.5
        durations.append(n * _UNIT_MONTHS[unit.lower()])
    if re.search(r"\bhalf\s+(?:a\s+)?year\b", text or "", re.I):
        durations.append(6)
    return max(durations) if durations else None


def interest_from_timeline(text: str) -> str | None:
    """hot = within 3 months, warm = 3 to 12 months, cold = more than 12 months or unsure."""
    months = timeline_months(text)
    if months is not None:
        return "hot" if months <= 3 else "warm" if months <= 12 else "cold"
    if _SOON.search(text or ""):
        return "hot"
    if _UNSURE.search(text or ""):
        return "cold"
    return None


def rule_changes(row: dict) -> dict:
    """The fields the rules would change on this CRM row (empty dict = nothing to change)."""
    changes = {}
    interest = interest_from_timeline(row.get("timeline") or "")
    if interest and interest != row.get("interest_level"):
        changes["interest_level"] = interest
    status = derive_status(row)
    if status and status != row.get("status"):
        changes["status"] = status
    return changes


def apply_rules(crm, lead_id: str) -> dict:
    changes = rule_changes(crm.get(lead_id))
    if changes:
        crm.update(lead_id, changes)
    return changes


# --- area ----------------------------------------------------------------------------------------

_VAGUE_AREA = re.compile(r"\b(any ?where|any area|any location|no preference|don'?t mind|doesn'?t matter|"
                         r"not sure|flexible|open to (?:any|all))\b", re.I)


def is_vague_area(text: str) -> bool:
    return bool(_VAGUE_AREA.search(text or ""))


# --- callback date and time ----------------------------------------------------------------------

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_PARTS = {"morning": (10, 0), "noon": (12, 0), "midday": (12, 0), "afternoon": (15, 0),
              "evening": (18, 0), "tonight": (19, 0), "night": (19, 0)}


def resolve_day(day: str, now: datetime) -> datetime:
    text = re.sub(r"\b(next|this|coming|on|the)\b", " ", (day or "").lower()).strip()
    text = re.sub(r"\s+", " ", text)
    if text in ("today", "tonight"):
        return now
    if text == "tomorrow":
        return now + timedelta(days=1)
    if text == "day after tomorrow":
        return now + timedelta(days=2)
    for i, name in enumerate(_WEEKDAYS):
        if text in (name, name[:3]):
            ahead = (i - now.weekday()) % 7 or 7  # always the next one after today
            return now + timedelta(days=ahead)
    try:
        return datetime.fromisoformat(text[:10])
    except ValueError:
        raise ValueError(f"day '{day}' not understood: use today, tomorrow, a weekday like monday, "
                         f"or a date like {now + timedelta(days=1):%Y-%m-%d}") from None


def resolve_time(time: str) -> tuple[int, int]:
    text = (time or "").lower().strip().replace(".", ":")
    for part, hm in _DAY_PARTS.items():
        if text == part or text == f"in the {part}":
            return hm
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|a:m:|p:m:)?", text)
    if not m:
        raise ValueError(f"time '{time}' not understood: use HH:MM, 24-hour, e.g. 16:00")
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    if m.group(3) and m.group(3).startswith("p") and hour < 12:
        hour += 12
    if m.group(3) and m.group(3).startswith("a") and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"time '{time}' is not a valid time of day")
    return hour, minute


def resolve_callback(day: str, time: str, now: datetime) -> datetime:
    """('monday', '10:00') on Wednesday 14 Oct 2026 -> 2026-10-19 10:00. Raises ValueError with a readable reason."""
    date = resolve_day(day, now)
    hour, minute = resolve_time(time)
    when = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if when <= now:
        raise ValueError(f"{when:%A %d %B %H:%M} is in the past (now is {now:%A %d %B %H:%M})")
    return when


# --- money guard ---------------------------------------------------------------------------------

SAFE_PRICE_LINE = "A property consultant will share the exact prices and details with you."
_MULTIPLIER = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6}
_MONEY_WORD_NUMBER = re.compile(
    r"\b(" + "|".join(k for k in _WORD_NUMBERS if k not in ("a", "an", "couple", "few", "several"))
    + r"|hundred)(?:\s+point\s+(" + "|".join(["zero", "one", "two", "three", "four", "five", "six", "seven",
                                               "eight", "nine"]) + r"))?(\s+and\s+a\s+half)?\s+(million|thousand)\b",
    re.I)
_MONEY_DIGITS = re.compile(r"(?:(aed|dhs?|dirhams?)\s*)?(\d+(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k|mn|m|million|thousand)?"
                           r"(?:\s*(aed|dhs?|dirhams?))?\b", re.I)
_DIGIT_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9}


def money_amounts(text: str) -> set[int]:
    """Amounts of money in a sentence: '2 million', 'AED 2,000,000', 'one point two million'."""
    found = set()
    for m in _MONEY_DIGITS.finditer(text or ""):
        before, number, mult, after = m.groups()
        value = float(number.replace(",", ""))
        if mult:
            value *= _MULTIPLIER[mult.lower()]
        elif not (before or after or "," in number or value >= 10000):
            continue  # a plain small number: a time, a count of bedrooms or months, a year
        found.add(int(round(value)))
    for m in _MONEY_WORD_NUMBER.finditer(text or ""):
        word, decimal, half, mult = m.groups()
        value = 100 if word.lower() == "hundred" else _WORD_NUMBERS[word.lower()]
        if decimal:
            value += _DIGIT_WORDS[decimal.lower()] / 10
        if half:
            value += 0.5
        found.add(int(round(value * _MULTIPLIER[mult.lower()])))
    return found


def guard_money(text: str, allowed: set[int]) -> tuple[str, list[str]]:
    """Remove sentences that state an amount of money the lead never said. Returns (text, removed)."""
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    kept, removed = [], []
    for s in sentences:
        amounts = money_amounts(s)
        if amounts and not amounts <= allowed:
            removed.append(s)
            if SAFE_PRICE_LINE not in kept:
                kept.append(SAFE_PRICE_LINE)
        else:
            kept.append(s)
    return (" ".join(k for k in kept if k).strip(), removed) if removed else (text, [])
