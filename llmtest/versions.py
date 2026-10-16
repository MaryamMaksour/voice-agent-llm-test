"""Which code and prompt produced a call, so results can be compared before and after a change."""

from __future__ import annotations

import hashlib
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))


def prompt_version() -> str:
    """First 10 characters of the SHA-256 of prompts.py: changes whenever the prompt text changes."""
    with open(os.path.join(_HERE, "prompts.py"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:10]


def code_version() -> str:
    """Short git commit of this checkout, or "unknown" (e.g. the folder was copied without .git)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_HERE, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
