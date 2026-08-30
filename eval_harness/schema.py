"""Pydantic schemas for the stratified evaluation dataset.

Each dataset record pairs an anonymized model *input* (one call) with a
human-labelled *ground truth*. Labels are keyed by stable identifiers — custom
fields and template extensions by **slug**, never by position — so the dataset
stays valid as templates evolve.

Pure data shapes plus small helpers; this module performs no DB or network I/O,
so it is cheap to import and unit-test.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

LeadQuality = Literal[
    "hot",
    "warm",
    "cold",
    "non_target",
    "repeat_active",
    "repeat_passive",
    "qualified",
    "partially_qualified",
    "not_qualified",
    "data_insufficient",
]
DurationBucket = Literal["short", "medium", "long", "very_long", "unknown"]
Split = Literal["train", "holdout"]

# The allowed values as plain tuples, for callers that iterate over them.
LEAD_QUALITY_VALUES: tuple[str, ...] = get_args(LeadQuality)
DURATION_BUCKETS: tuple[str, ...] = get_args(DurationBucket)


def duration_bucket(duration_sec: int | None) -> DurationBucket:
    """Map a call duration in seconds to a coarse bucket; ``None`` -> ``"unknown"``.

    The <60s boundary separates calls that are usually too short to carry an
    analysis payload (missed calls, wrong numbers, spam).
    """
    if duration_sec is None:
        return "unknown"
    if duration_sec < 60:
        return "short"
    if duration_sec < 180:
        return "medium"
    if duration_sec < 600:
        return "long"
    return "very_long"


class SpeakerTurn(BaseModel):
    """One anonymized speaker turn from the transcript."""

    speaker: str
    text: str


class CallInput(BaseModel):
    """Anonymized model input for one call — no PII, safe to hand to labellers."""

    call_id: int
    company_id: int
    template_id: int | None = None
    template_name: str | None = None
    department_id: int | None = None
    duration_sec: int | None = None
    duration_bucket: DurationBucket = "unknown"
    transcript: str = ""
    speaker_turns: list[SpeakerTurn] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_duration_bucket(self) -> CallInput:
        # A known duration always wins over a stale or missing bucket.
        if self.duration_sec is not None:
            self.duration_bucket = duration_bucket(self.duration_sec)
        return self


class StratumKey(BaseModel):
    """The tuple a record is stratified on.

    ``industry`` is intentionally absent: it is rarely available as structured
    data, so it is captured as an optional human-labelled field on
    :class:`CallLabel` rather than used as a sampling axis.
    """

    company_id: int
    template_id: int | None = None
    call_type_key: str | None = None
    duration_bucket: DurationBucket = "unknown"

    def as_tuple(self) -> tuple[int, int | None, str | None, str]:
        return (self.company_id, self.template_id, self.call_type_key, self.duration_bucket)


class CallLabel(BaseModel):
    """Human-provided ground truth for one call. Empty until a labeller saves it."""

    call_type_key: str | None = None
    lead_quality: LeadQuality | None = None
    custom_fields: dict[str, str] = Field(default_factory=dict)  # keyed by checklist-field slug
    extensions: dict[str, Any] = Field(default_factory=dict)  # keyed by template-extension slug
    risk_category_keys: list[str] = Field(default_factory=list)
    summary_quality: int | None = Field(default=None, ge=1, le=5)  # 1-5 Likert
    industry: str | None = None  # optional, human-labelled
    labelled_by: str | None = None
    labelled_at: str | None = None  # ISO-8601; set only when a human saves
    notes: str | None = None

    @field_validator("labelled_at")
    @classmethod
    def _check_labelled_at(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"labelled_at must be an ISO-8601 timestamp, got {value!r}") from exc
        return value

    @property
    def is_labelled(self) -> bool:
        """True once a human has saved the label (pre-filled drafts stay False)."""
        return self.labelled_at is not None


class EvalRecord(BaseModel):
    """One JSONL line: anonymized input + label + stratum + train/holdout split."""

    input: CallInput
    label: CallLabel = Field(default_factory=CallLabel)
    stratum: StratumKey
    split: Split | None = None
