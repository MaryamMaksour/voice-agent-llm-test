"""The leads 'CRM' is a CSV file. Every change is validated and saved immediately."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from datetime import datetime

FIELDS = [
    "lead_id", "name", "phone", "project_interest", "status", "budget_aed", "preferred_area",
    "property_type", "timeline", "interest_level", "callback_at", "do_not_call", "notes", "updated_at",
]
STATUSES = ["new", "contacted", "interested", "qualified", "not_interested", "callback",
            "do_not_call", "wrong_number"]
PROPERTY_TYPES = ["apartment", "villa", "townhouse", "penthouse", "office", "land"]
INTEREST_LEVELS = ["hot", "warm", "cold"]
EDITABLE = ["status", "budget_aed", "preferred_area", "property_type", "timeline", "interest_level"]


class CRMError(ValueError):
    pass


def validate_changes(changes: dict) -> dict:
    """Return cleaned changes, or raise CRMError with a message the model can act on."""
    clean = {}
    for key, value in changes.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if key not in EDITABLE:
            raise CRMError(f"'{key}' is not a field you can update")
        if key == "status":
            value = str(value).strip().lower()
            if value not in STATUSES:
                raise CRMError(f"status must be one of {STATUSES}")
        elif key == "property_type":
            value = str(value).strip().lower()
            if value not in PROPERTY_TYPES:
                raise CRMError(f"property_type must be one of {PROPERTY_TYPES}")
        elif key == "interest_level":
            value = str(value).strip().lower()
            if value not in INTEREST_LEVELS:
                raise CRMError(f"interest_level must be one of {INTEREST_LEVELS}")
        elif key == "budget_aed":
            try:
                value = int(float(str(value).replace(",", "")))
            except ValueError:
                raise CRMError("budget_aed must be a whole number of dirhams, e.g. 2000000") from None
            if value <= 0:
                raise CRMError("budget_aed must be positive")
        else:
            value = str(value).strip()
        clean[key] = value
    return clean


class CsvCRM:
    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, newline="", encoding="utf-8") as f:
            self.rows = {r["lead_id"]: r for r in csv.DictReader(f)}

    @classmethod
    def fresh_copy(cls, seed_path: str, dest_path: str) -> "CsvCRM":
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        shutil.copyfile(seed_path, dest_path)
        return cls(dest_path)

    def get(self, lead_id: str) -> dict:
        if lead_id not in self.rows:
            raise CRMError(f"lead {lead_id} not found")
        return dict(self.rows[lead_id])

    def _write(self, lead_id: str, changes: dict) -> dict:
        row = self.rows[lead_id]
        row.update({k: str(v) for k, v in changes.items()})
        row["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.save()
        return dict(row)

    def update(self, lead_id: str, changes: dict) -> dict:
        self.get(lead_id)
        return self._write(lead_id, validate_changes(changes))

    def schedule_callback(self, lead_id: str, when: datetime) -> dict:
        self.get(lead_id)
        return self._write(lead_id, {"callback_at": when.strftime("%Y-%m-%dT%H:%M"), "status": "callback"})

    def mark_do_not_call(self, lead_id: str) -> dict:
        self.get(lead_id)
        return self._write(lead_id, {"do_not_call": "true", "status": "do_not_call"})

    def add_note(self, lead_id: str, note: str) -> dict:
        row = self.get(lead_id)
        notes = f"{row['notes']} | {note}" if row.get("notes") else note
        return self._write(lead_id, {"notes": notes})

    def save(self) -> None:
        """Atomic write: never leaves a half-written CSV."""
        folder = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for row in self.rows.values():
                writer.writerow({k: row.get(k, "") for k in FIELDS})
        os.replace(tmp, self.path)
