"""Cost-estimation constants for benchmark reports.

Approximate USD list prices per 1K tokens for a cross-provider benchmark matrix.
Best-effort — MUST be re-verified at benchmark time (provider pricing changes).
These feed only a report's cost column, never billing or production tracing.
"""

from __future__ import annotations

from typing import Any

PRICES_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5": {"input": 0.001, "output": 0.005},
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
}


def estimate_cost_usd(model: str | None, usage: Any) -> float | None:
    """USD cost for one call from token usage + the approximate price table.

    ``usage`` is any object exposing ``prompt_tokens`` / ``completion_tokens``
    (the OpenAI-style usage shape). Returns ``None`` when the model is unknown
    or usage is incomplete — a missing cost must never break a report.
    """
    prices = PRICES_PER_1K.get(model or "")
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prices is None or prompt_tokens is None or completion_tokens is None:
        return None
    return float(prompt_tokens) / 1000.0 * prices["input"] + float(completion_tokens) / 1000.0 * prices["output"]
