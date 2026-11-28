"""Input guardrails (length limit, prompt-injection patterns) and slowapi rate limiter."""

from __future__ import annotations

import re

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|system)\s+(rules|prompts?)", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(DAN|jailbroken|unrestricted)", re.IGNORECASE),
    re.compile(r"</\s*(retrieved_context|sql_results|db_schema)\s*>", re.IGNORECASE),
]


class InputGuardrailError(ValueError):
    """Raised when user input violates length or prompt-injection guardrails."""


def validate_user_query(query: str, *, max_chars: int = 2000) -> str:
    cleaned = query.strip()
    if not cleaned:
        raise InputGuardrailError("Query must not be empty.")
    if len(cleaned) > max_chars:
        raise InputGuardrailError(
            f"Query length ({len(cleaned)}) exceeds maximum allowed {max_chars} characters."
        )
    for pattern in INJECTION_PATTERNS:
        if pattern.search(cleaned):
            raise InputGuardrailError(
                "Query rejected by input guardrail: detected potential prompt-injection pattern."
            )
    return cleaned
