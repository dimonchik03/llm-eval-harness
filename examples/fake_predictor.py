"""Deterministic offline predictor for the demo — NOT a model.

Replays each record's ground-truth label and injects deterministic, hash-seeded
noise (a wrong call type here, a shifted lead quality there) so the demo report
shows realistic, non-trivial metrics with zero network access and zero API keys.
Because it peeks at the label, its scores mean nothing — it exists purely so
``make demo`` exercises the full harness path. To evaluate a real model, provide
your own ``predict_fn(record) -> StagePrediction`` that calls your pipeline (see
the README's "Plugging in your model" section).
"""

from __future__ import annotations

import hashlib

from eval_harness import LEAD_QUALITY_VALUES, EvalRecord, StagePrediction

MODEL_NAME = "fake-offline-v0"

# Call types whose calls carry no analytical payload. A typical pipeline skips
# the main analysis for these, and so does this stub.
NON_ANALYTICAL_TYPES = frozenset({"spam", "voicemail", "wrong_number"})

_ANALYTICAL_TYPES = ("sales_consultation", "support_request")


def _bucket(record: EvalRecord, salt: str, mod: int = 100) -> int:
    """Stable pseudo-random 0..mod-1 value per (record, salt) — same on every run."""
    digest = hashlib.sha256(f"{salt}:{record.input.call_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % mod


def predict(record: EvalRecord) -> StagePrediction:
    """Label replay + deterministic noise; mirrors the shape of a real pipeline adapter."""
    label = record.label
    prediction = StagePrediction(
        call_id=record.input.call_id,
        company_id=record.input.company_id,
        template_id=record.input.template_id,
        industry=label.industry,
        model=MODEL_NAME,
        latency_ms=35.0 + _bucket(record, "latency", 40),
    )

    true_type = label.call_type_key or "sales_consultation"
    prediction.call_type_key = true_type
    if _bucket(record, "call_type") < 10:  # ~10% wrong call type
        prediction.call_type_key = next(t for t in _ANALYTICAL_TYPES if t != true_type)
    prediction.is_analytical = prediction.call_type_key not in NON_ANALYTICAL_TYPES
    if not prediction.is_analytical:
        return prediction

    lead = label.lead_quality
    if lead is not None and _bucket(record, "lead") < 15:  # ~15% off-by-one lead quality
        index = LEAD_QUALITY_VALUES.index(lead)
        lead = LEAD_QUALITY_VALUES[(index + 1) % len(LEAD_QUALITY_VALUES)]
    prediction.lead_quality = lead

    risks = list(label.risk_category_keys)
    if risks and _bucket(record, "risk_drop") < 20:  # ~20% miss one risk
        risks = risks[:-1]
    if _bucket(record, "risk_add") < 10 and "delivery_concern" not in risks:  # ~10% hallucinate one
        risks.append("delivery_concern")
    prediction.risk_category_keys = risks

    fields = dict(label.custom_fields)
    if fields and _bucket(record, "field") < 15:  # ~15% garble one custom-field value
        slug = sorted(fields)[0]
        fields[slug] = f"{fields[slug]} (approx.)"
    prediction.custom_fields = fields
    prediction.extensions = dict(label.extensions)

    first_client_line = next(
        (turn.text for turn in record.input.speaker_turns if turn.speaker == "client"),
        "",
    )
    prediction.summary = f"Client contacted the company. Opening request: {first_client_line}".strip()
    return prediction
