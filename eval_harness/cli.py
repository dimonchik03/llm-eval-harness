"""``llm-eval`` — run the harness over a dataset and write a JSON quality report.

Examples::

    llm-eval --dataset examples/synthetic_demo.jsonl --predictor examples.fake_predictor:predict
    llm-eval --dataset data/dataset_v1.jsonl --split holdout --limit 50 \\
        --predictor my_pipeline.eval_adapter:predict --judge --concurrency 8 \\
        --out report.json --predictions-out predictions.jsonl --fail-on-regression 0.02

``--predictor`` names the ``predict_fn(record) -> StagePrediction`` seam as a
``package.module:function`` import path; the current working directory is put on
``sys.path`` so a module next to the dataset resolves without packaging.

Exit codes: 0 — run finished; 1 — ``--fail-on-regression`` tripped; 2 — bad
arguments or a missing dataset.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from eval_harness import __version__
from eval_harness.dataset_io import read_records
from eval_harness.runner import (
    DEFAULT_JUDGE_MODEL,
    PredictFn,
    build_report,
    compare_to_baseline,
    evaluate,
    find_regressions,
    make_faithfulness_judge,
    prediction_rows,
    print_headline,
    select_records,
)

DEFAULT_BASELINE = "eval_baseline.json"
EXIT_OK, EXIT_REGRESSION, EXIT_USAGE = 0, 1, 2


def _load_predictor(spec: str) -> PredictFn:
    """Resolve a ``package.module:function`` spec to the predict_fn callable."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise SystemExit(f"--predictor must look like 'package.module:function', got {spec!r}")
    module = importlib.import_module(module_name)
    predict_fn = getattr(module, attr, None)
    if not callable(predict_fn):
        raise SystemExit(f"--predictor {spec!r} does not resolve to a callable")
    return predict_fn  # type: ignore[no-any-return]


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: str, rows: Sequence[Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="llm-eval", description="Run the LLM-analysis eval harness.")
    parser.add_argument("--dataset", required=True, help="JSONL eval dataset.")
    parser.add_argument("--predictor", required=True, help="predict_fn import path, e.g. 'my_pkg.adapter:predict'.")
    parser.add_argument("--split", choices=["train", "holdout"], default=None, help="Restrict to a split.")
    parser.add_argument("--limit", type=_positive_int, default=None, help="Cap evaluated records (CI fast subset).")
    parser.add_argument(
        "--include-unlabelled", action="store_true", help="Also evaluate records without a human label."
    )
    parser.add_argument("--judge", action="store_true", help="Run the summary-faithfulness LLM judge.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Fixed judge model (default: %(default)s).")
    parser.add_argument("--concurrency", type=_positive_int, default=1, help="Records evaluated in parallel.")
    parser.add_argument("--out", default=None, help="Write the full JSON report to this path.")
    parser.add_argument("--predictions-out", default=None, help="Write one JSONL row per record (truth vs pred).")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="Baseline report (default: %(default)s).")
    parser.add_argument(
        "--update-baseline", action="store_true", help="Overwrite the baseline file with this run's report."
    )
    parser.add_argument(
        "--fail-on-regression",
        type=float,
        default=None,
        metavar="DELTA",
        help="Exit 1 when a headline metric drops by more than DELTA (absolute) vs the baseline.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def _load_baseline(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        baseline = json.load(handle)
    return baseline if isinstance(baseline, dict) else {}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.WARNING, format="[eval] %(levelname)s: %(message)s")

    if not os.path.exists(args.dataset):
        print(f"[eval] dataset not found: {args.dataset}", file=sys.stderr)
        return EXIT_USAGE

    sys.path.insert(0, os.getcwd())  # let --predictor name a module relative to the invocation dir
    predict_fn = _load_predictor(args.predictor)
    judge = make_faithfulness_judge(args.judge_model) if args.judge else None

    records = select_records(
        read_records(args.dataset),
        split=args.split,
        labelled_only=not args.include_unlabelled,
        limit=args.limit,
    )
    metadata = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "harness_version": __version__,
        "predictor": args.predictor,
        "judge_model": args.judge_model if args.judge else None,
        "labelled_only": not args.include_unlabelled,
        "limit": args.limit,
        "concurrency": args.concurrency,
    }
    pairs = evaluate(records, predict_fn, judge=judge, concurrency=args.concurrency)
    report = build_report(pairs, dataset_path=args.dataset, split=args.split, metadata=metadata)

    baseline = _load_baseline(args.baseline)
    regressions: dict[str, float] = {}
    if baseline.get("overall"):
        report["baseline_delta"] = compare_to_baseline(report, baseline)
        if args.fail_on_regression is not None:
            regressions = find_regressions(report["baseline_delta"], args.fail_on_regression)
    elif args.fail_on_regression is not None:
        print(f"[eval] no baseline at {args.baseline} — regression gate skipped", file=sys.stderr)

    if args.out:
        _write_json(args.out, report)
        print(f"[eval] wrote report to {args.out}")
    if args.predictions_out:
        _write_jsonl(args.predictions_out, prediction_rows(pairs))
        print(f"[eval] wrote {len(pairs)} prediction row(s) to {args.predictions_out}")
    if args.update_baseline:
        _write_json(args.baseline, report)
        print(f"[eval] updated baseline at {args.baseline}")

    print_headline(report)

    if regressions:
        details = ", ".join(f"{key} {value:+.3f}" for key, value in regressions.items())
        print(f"[eval] REGRESSION beyond {args.fail_on_regression}: {details}", file=sys.stderr)
        return EXIT_REGRESSION
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
