"""Eval harness runner.

Drives a caller-supplied prediction function over a labelled eval dataset and
collects (record, prediction) pairs, then computes per-stage / per-company /
per-template / per-industry quality metrics via :mod:`.metrics`.

The harness owns no model. :func:`run_eval` takes ``predict_fn(record) ->
StagePrediction`` — the seam where callers plug in their pipeline, a candidate
prompt, a different provider, or a deterministic stub (see
``examples/fake_predictor.py``).

Failure handling is explicit, never silent:

- A prediction that raises is captured as an ``error`` on that record, so one bad
  record never aborts a run. Errored records are excluded from the quality
  metrics and reported separately as ``error_rate`` — a provider outage must not
  masquerade as a quality regression.
- The optional LLM-as-judge fails closed: a refusal, an unparseable answer or an
  out-of-range score becomes a ``judge_error`` on the record and is never counted
  as a faithful summary.

The default judge calls OpenAI structured outputs via ``OPENAI_API_KEY`` (install
the ``judge`` extra); its single LLM seam, :func:`_call_judge_llm`, is
monkeypatched in tests so no key or network is needed. Callers may also pass any
``judge`` callable of their own.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TextIO

from pydantic import BaseModel, field_validator

from eval_harness import metrics
from eval_harness.metrics import normalize
from eval_harness.schema import EvalRecord, Split

logger = logging.getLogger("eval_harness")

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


@dataclass
class StagePrediction:
    """The pipeline's output for one record, reduced to the labelled surface."""

    call_id: int
    company_id: int
    template_id: int | None = None
    industry: str | None = None
    call_type_key: str | None = None
    is_analytical: bool | None = None
    lead_quality: str | None = None
    custom_fields: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)
    risk_category_keys: list[str] = field(default_factory=list)
    summary: str | None = None
    faithfulness_score: int | None = None
    faithful: bool | None = None
    judge_error: str | None = None
    model: str | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    error: str | None = None


class FaithfulnessVerdict(BaseModel):
    """Structured verdict from the summary-faithfulness judge (LLM-as-judge).

    Every field is required: a verdict the judge did not actually produce must
    never default to "faithful".
    """

    faithful: bool
    score: int  # 1 (mostly fabricated) .. 5 (fully grounded)
    unsupported_claims: list[str]

    @field_validator("score")
    @classmethod
    def _check_score(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError(f"score must be between 1 and 5, got {value}")
        return value


class JudgeError(RuntimeError):
    """The judge produced no usable verdict (refusal, empty or unparseable output)."""


# The model-under-test seam: map one dataset record to a prediction.
PredictFn = Callable[[EvalRecord], StagePrediction]

# A judge takes (transcript, narrative) and returns a verdict. It may raise; the
# runner records that as a judge_error on the record.
Judge = Callable[[str, str], FaithfulnessVerdict]

Pair = tuple[EvalRecord, StagePrediction]

_JUDGE_PROMPT = (
    "You are a strict evaluator of sales-call analysis. Given the call TRANSCRIPT and a "
    "generated ANALYSIS NARRATIVE, find every claim in the narrative that is NOT supported "
    "by the transcript. Return faithful=false if any material claim is unsupported, a 1-5 "
    "score (5 = fully grounded in the transcript, 1 = mostly fabricated), and the list of "
    "unsupported claims.\n\nANALYSIS NARRATIVE:\n{narrative}"
)


def _call_judge_llm(transcript: str, prompt: str, model: str) -> FaithfulnessVerdict | None:
    """The judge's one LLM call — OpenAI structured outputs, key from the environment.

    Returns ``None`` when the model refuses or returns nothing parseable. Lazily
    imports ``openai`` so the package works without the ``judge`` extra
    installed; tests monkeypatch this function instead of the network.
    """
    from openai import OpenAI  # lazy: only needed when actually judging

    client = OpenAI()
    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
        response_format=FaithfulnessVerdict,
    )
    parsed: FaithfulnessVerdict | None = completion.choices[0].message.parsed
    return parsed


def make_faithfulness_judge(model: str = DEFAULT_JUDGE_MODEL) -> Judge:
    """Build the LLM-as-judge faithfulness check around a FIXED judge model.

    The judge model is independent of the model under test: it must stay a
    constant evaluator across a cross-provider benchmark, so pin it per benchmark.
    """

    def judge(transcript: str, narrative: str) -> FaithfulnessVerdict:
        verdict = _call_judge_llm(transcript, _JUDGE_PROMPT.format(narrative=narrative), model)
        if verdict is None:
            raise JudgeError(f"judge model {model!r} returned no parsed verdict (refusal or empty output)")
        return verdict

    return judge


default_faithfulness_judge: Judge = make_faithfulness_judge()


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _safe_predict(record: EvalRecord, predict_fn: PredictFn) -> StagePrediction:
    """Call ``predict_fn``; a raised exception becomes an ``error`` on the prediction."""
    try:
        prediction = predict_fn(record)
        if not isinstance(prediction, StagePrediction):
            raise TypeError(f"predict_fn must return StagePrediction, got {type(prediction).__name__}")
        return prediction
    except Exception as exc:  # noqa: BLE001 — the harness must survive a single bad record
        logger.warning("prediction failed for call %s: %s", record.input.call_id, _describe(exc))
        return StagePrediction(
            call_id=record.input.call_id,
            company_id=record.input.company_id,
            template_id=record.input.template_id,
            industry=record.label.industry,
            error=_describe(exc),
        )


def _should_judge(prediction: StagePrediction) -> bool:
    # ``is_analytical`` unset (None) still gets judged; only an explicit False skips.
    return prediction.error is None and bool(prediction.summary) and prediction.is_analytical is not False


def _apply_judge(record: EvalRecord, prediction: StagePrediction, judge: Judge) -> None:
    """Judge one summary; any failure is recorded as ``judge_error`` and never scored."""
    try:
        verdict = judge(record.input.transcript or "", prediction.summary or "")
        score = int(verdict.score)
        if not 1 <= score <= 5:
            raise JudgeError(f"judge score out of range 1-5: {score}")
        prediction.faithfulness_score = score
        prediction.faithful = bool(verdict.faithful)
    except Exception as exc:  # noqa: BLE001 — judging is best-effort, never fatal
        prediction.judge_error = _describe(exc)
        logger.warning("judge failed for call %s: %s", prediction.call_id, prediction.judge_error)


def select_records(
    records: Iterable[EvalRecord],
    *,
    split: Split | None = None,
    labelled_only: bool = True,
    limit: int | None = None,
) -> list[EvalRecord]:
    """Filter to a split and to human-labelled records, then cap the count."""
    selected = [record for record in records if split is None or record.split == split]
    if labelled_only:
        selected = [record for record in selected if record.label.is_labelled]
    return selected[:limit] if limit is not None else selected


def evaluate(
    records: Sequence[EvalRecord],
    predict_fn: PredictFn,
    *,
    judge: Judge | None = None,
    concurrency: int = 1,
) -> list[Pair]:
    """Predict (and optionally judge) every record; returns pairs in input order.

    ``concurrency`` > 1 runs records on a thread pool — LLM calls are I/O-bound,
    so threads are enough. ``predict_fn`` and ``judge`` must then be thread-safe.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    def one(record: EvalRecord) -> Pair:
        prediction = _safe_predict(record, predict_fn)
        if judge is not None and _should_judge(prediction):
            _apply_judge(record, prediction, judge)
        return record, prediction

    if concurrency == 1:
        return [one(record) for record in records]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, records))


def run_eval(
    records: Sequence[EvalRecord],
    predict_fn: PredictFn,
    *,
    judge: Judge | None = None,
    limit: int | None = None,
    split: Split | None = None,
    labelled_only: bool = True,
    concurrency: int = 1,
    dataset_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the harness over ``records`` and return a JSON-serializable report.

    ``labelled_only`` (default) skips records a human has not yet labelled — there is no
    metric value in predicting them and it wastes LLM budget. ``split`` filters to a
    train/holdout slice; ``limit`` caps the count (the CI fast subset).
    """
    selected = select_records(records, split=split, labelled_only=labelled_only, limit=limit)
    pairs = evaluate(selected, predict_fn, judge=judge, concurrency=concurrency)
    n_errors = sum(1 for _, prediction in pairs if prediction.error)
    logger.info("evaluated %d record(s); %d error(s)", len(pairs), n_errors)
    return build_report(pairs, dataset_path=dataset_path, split=split, metadata=metadata)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _faithfulness(pairs: Sequence[Pair]) -> dict[str, Any]:
    scores = [pred.faithfulness_score for _, pred in pairs if pred.faithfulness_score is not None]
    flags = [pred.faithful for _, pred in pairs if pred.faithful is not None]
    return {
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "faithful_rate": (sum(1 for flag in flags if flag) / len(flags)) if flags else None,
        "faithful_rate_ci95": metrics.wilson_interval(sum(1 for flag in flags if flag), len(flags)),
        "support": len(scores),
        "n_judge_errors": sum(1 for _, pred in pairs if pred.judge_error),
    }


def _percentile(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile; ``None`` for an empty sequence."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def _latency_cost(pairs: Sequence[Pair]) -> dict[str, Any]:
    """Per-call latency (mean/p50/p95) + cost aggregates, when the predictor reports them."""
    latencies = [pred.latency_ms for _, pred in pairs if pred.latency_ms is not None]
    costs = [pred.cost_usd for _, pred in pairs if pred.cost_usd is not None]
    return {
        "models": sorted({pred.model for _, pred in pairs if pred.model}),
        "latency_ms": {
            "mean": (sum(latencies) / len(latencies)) if latencies else None,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "support": len(latencies),
        },
        "cost_usd": {
            "total": sum(costs) if costs else None,
            "per_call_mean": (sum(costs) / len(costs)) if costs else None,
            "support": len(costs),
        },
    }


def _metrics_block(pairs: Sequence[Pair]) -> dict[str, Any]:
    """All per-stage metrics for one slice; quality metrics cover successful predictions only."""
    scored = [(rec, pred) for rec, pred in pairs if pred.error is None]
    n_errors = len(pairs) - len(scored)
    call_type_pairs = [(rec.label.call_type_key, pred.call_type_key) for rec, pred in scored]
    block: dict[str, Any] = {
        "n": len(pairs),
        "n_evaluated": len(scored),
        "n_errors": n_errors,
        "error_rate": (n_errors / len(pairs)) if pairs else None,
        "call_type": {
            **metrics.accuracy(call_type_pairs),
            **metrics.per_class_prf(call_type_pairs),
            "confusion": metrics.confusion_matrix(call_type_pairs),
        },
        "lead_quality": metrics.accuracy([(rec.label.lead_quality, pred.lead_quality) for rec, pred in scored]),
        "risks": metrics.multilabel_prf(
            [(rec.label.risk_category_keys, pred.risk_category_keys) for rec, pred in scored]
        ),
        "custom_fields": metrics.keyed_value_prf(
            [(rec.label.custom_fields, pred.custom_fields) for rec, pred in scored]
        ),
        "extensions": metrics.keyed_value_prf([(rec.label.extensions, pred.extensions) for rec, pred in scored]),
        "performance": _latency_cost(scored),
    }
    if any(pred.faithfulness_score is not None or pred.judge_error for _, pred in scored):
        block["summary_faithfulness"] = _faithfulness(scored)
    return block


def build_report(
    pairs: Sequence[Pair],
    *,
    dataset_path: str | None = None,
    split: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the overall + per-company / per-template / per-industry report."""
    return {
        "run": dict(metadata or {}),
        "dataset": {
            "path": dataset_path,
            "split": split,
            "n_records": len(pairs),
            "n_evaluated": sum(1 for _, pred in pairs if pred.error is None),
            "n_errors": sum(1 for _, pred in pairs if pred.error),
        },
        "overall": _metrics_block(pairs),
        "by_company": metrics.bucketize(pairs, lambda rp: rp[0].input.company_id, _metrics_block),
        "by_template": metrics.bucketize(pairs, lambda rp: rp[0].input.template_id, _metrics_block),
        # industry is a human-labelled field, so by_industry stays empty unless
        # labellers fill it in.
        "by_industry": metrics.bucketize(pairs, lambda rp: rp[0].label.industry, _metrics_block),
        "errors": [{"call_id": pred.call_id, "error": pred.error} for _, pred in pairs if pred.error],
    }


def _single_label_row(truth: Any, pred: Any) -> dict[str, Any]:
    expected = normalize(truth)
    return {"truth": truth, "pred": pred, "match": None if expected is None else expected == normalize(pred)}


def _keyed_mismatches(truth: Mapping[str, Any], pred: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    slugs = {k for k, v in truth.items() if normalize(v) is not None} | {
        k for k, v in pred.items() if normalize(v) is not None
    }
    return {
        slug: {"truth": truth.get(slug), "pred": pred.get(slug)}
        for slug in sorted(slugs)
        if normalize(truth.get(slug)) != normalize(pred.get(slug))
    }


def prediction_rows(pairs: Sequence[Pair]) -> list[dict[str, Any]]:
    """One JSON-serializable row per record: truth vs prediction and what went wrong.

    The aggregate report says *how much* changed; these rows say *which calls* —
    write them with ``--predictions-out`` and filter on ``all_correct``.
    """
    rows: list[dict[str, Any]] = []
    for record, pred in pairs:
        label = record.label
        truth_risks = {n for x in label.risk_category_keys if (n := normalize(x)) is not None}
        pred_risks = {n for x in pred.risk_category_keys if (n := normalize(x)) is not None}
        row: dict[str, Any] = {
            "call_id": record.input.call_id,
            "company_id": record.input.company_id,
            "template_id": record.input.template_id,
            "split": record.split,
            "error": pred.error,
            "call_type": _single_label_row(label.call_type_key, pred.call_type_key),
            "lead_quality": _single_label_row(label.lead_quality, pred.lead_quality),
            "risks": {"missing": sorted(truth_risks - pred_risks), "extra": sorted(pred_risks - truth_risks)},
            "custom_fields": _keyed_mismatches(label.custom_fields, pred.custom_fields),
            "extensions": _keyed_mismatches(label.extensions, pred.extensions),
            "faithful": pred.faithful,
            "faithfulness_score": pred.faithfulness_score,
            "judge_error": pred.judge_error,
            "model": pred.model,
            "latency_ms": pred.latency_ms,
            "cost_usd": pred.cost_usd,
        }
        row["all_correct"] = (
            pred.error is None
            and row["call_type"]["match"] is not False
            and row["lead_quality"]["match"] is not False
            and not row["risks"]["missing"]
            and not row["risks"]["extra"]
            and not row["custom_fields"]
            and not row["extensions"]
        )
        rows.append(row)
    return rows


def _headline(block: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "call_type_accuracy": (block.get("call_type") or {}).get("accuracy"),
        "lead_quality_accuracy": (block.get("lead_quality") or {}).get("accuracy"),
        "custom_fields_micro_f1": ((block.get("custom_fields") or {}).get("micro") or {}).get("f1"),
        "risks_micro_f1": ((block.get("risks") or {}).get("micro") or {}).get("f1"),
    }


def compare_to_baseline(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Headline metric deltas (report - baseline) for the overall block. ``None``-safe."""
    current = _headline(report.get("overall") or {})
    prior = _headline(baseline.get("overall") or {})
    overall_delta: dict[str, float | None] = {}
    for key, value in current.items():
        previous = prior[key]
        overall_delta[key] = value - previous if value is not None and previous is not None else None
    return {"overall": overall_delta}


def find_regressions(delta: Mapping[str, Any], threshold: float) -> dict[str, float]:
    """Headline metrics that dropped by more than ``threshold`` (absolute) vs the baseline."""
    overall = delta.get("overall") or {}
    return {key: value for key, value in overall.items() if value is not None and value < -threshold}


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _fmt_accuracy(block: Mapping[str, Any]) -> str:
    ci = block.get("ci95")
    return f"{_fmt(block.get('accuracy'))} [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else _fmt(block.get("accuracy"))


def _headline_line(name: str, block: Mapping[str, Any]) -> str:
    parts = [
        f"{name:<12} n={block.get('n_evaluated')!s:<3}",
        f"call_type {_fmt_accuracy(block.get('call_type') or {})}",
        f"lead_quality {_fmt_accuracy(block.get('lead_quality') or {})}",
        f"fields_f1 {_fmt(_headline(block)['custom_fields_micro_f1'])}",
        f"risks_f1 {_fmt(_headline(block)['risks_micro_f1'])}",
    ]
    if block.get("n_errors"):
        parts.append(f"errors {block['n_errors']}")
    return "[eval] " + "  ".join(parts)


def print_headline(report: Mapping[str, Any], file: TextIO | None = None) -> None:
    """Compact console summary: counts, overall + per-company headline metrics with 95% CIs."""
    out = file or sys.stdout
    dataset = report.get("dataset") or {}
    print(
        f"[eval] dataset={dataset.get('path')} split={dataset.get('split')} "
        f"records={dataset.get('n_records')} evaluated={dataset.get('n_evaluated')} "
        f"errors={dataset.get('n_errors')}",
        file=out,
    )
    overall = report.get("overall") or {}
    print(_headline_line("overall", overall), file=out)
    for company_id, block in (report.get("by_company") or {}).items():
        print(_headline_line(f"company {company_id}", block), file=out)
    faithfulness = overall.get("summary_faithfulness")
    if faithfulness:
        print(
            f"[eval] judge: faithful_rate {_fmt(faithfulness['faithful_rate'])}  "
            f"mean_score {_fmt(faithfulness['mean_score'])}  n={faithfulness['support']}  "
            f"judge_errors {faithfulness['n_judge_errors']}",
            file=out,
        )
    delta = (report.get("baseline_delta") or {}).get("overall") or {}
    if any(value is not None for value in delta.values()):
        changes = "  ".join(f"{key} {value:+.3f}" for key, value in delta.items() if value is not None)
        print(f"[eval] Δ vs baseline: {changes}", file=out)
