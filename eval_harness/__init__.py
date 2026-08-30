"""Evaluation harness for structured LLM outputs (call analytics).

Builds stratified, anonymized JSONL datasets of calls and measures analysis
quality per company / template / call-type: pure-Python metrics with confidence
intervals, an optional fail-closed LLM-as-judge faithfulness check, per-record
results and baseline deltas. The model under test is supplied by the caller as
``predict_fn(record) -> StagePrediction`` — the harness itself never talks to a
database or a provider.
"""

from eval_harness import metrics
from eval_harness.anonymize import anonymize_text, mask_patterns, redact_names
from eval_harness.constants import PRICES_PER_1K, estimate_cost_usd
from eval_harness.dataset_io import iter_records, read_records, write_records
from eval_harness.runner import (
    FaithfulnessVerdict,
    Judge,
    JudgeError,
    PredictFn,
    StagePrediction,
    build_report,
    compare_to_baseline,
    default_faithfulness_judge,
    evaluate,
    find_regressions,
    make_faithfulness_judge,
    prediction_rows,
    print_headline,
    run_eval,
    select_records,
)
from eval_harness.sampling import stratified_sample, stratified_split
from eval_harness.schema import (
    DURATION_BUCKETS,
    LEAD_QUALITY_VALUES,
    CallInput,
    CallLabel,
    EvalRecord,
    SpeakerTurn,
    StratumKey,
    duration_bucket,
)

__version__ = "0.1.0"

__all__ = [
    "DURATION_BUCKETS",
    "LEAD_QUALITY_VALUES",
    "PRICES_PER_1K",
    "CallInput",
    "CallLabel",
    "EvalRecord",
    "FaithfulnessVerdict",
    "Judge",
    "JudgeError",
    "PredictFn",
    "SpeakerTurn",
    "StagePrediction",
    "StratumKey",
    "__version__",
    "anonymize_text",
    "build_report",
    "compare_to_baseline",
    "default_faithfulness_judge",
    "duration_bucket",
    "estimate_cost_usd",
    "evaluate",
    "find_regressions",
    "iter_records",
    "make_faithfulness_judge",
    "mask_patterns",
    "metrics",
    "prediction_rows",
    "print_headline",
    "read_records",
    "redact_names",
    "run_eval",
    "select_records",
    "stratified_sample",
    "stratified_split",
    "write_records",
]
