"""CLI tests over the synthetic demo dataset — exit codes, outputs, baseline gate."""

import json
import os

from eval_harness.cli import EXIT_OK, EXIT_REGRESSION, EXIT_USAGE, main

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(REPO_ROOT, "examples", "synthetic_demo.jsonl")
PREDICTOR = "examples.fake_predictor:predict"


def _run(tmp_path, *extra):
    return main(["--dataset", DEMO, "--predictor", PREDICTOR, "--baseline", str(tmp_path / "baseline.json"), *extra])


def test_missing_dataset_is_a_usage_error(capsys):
    assert main(["--dataset", "/nonexistent/eval.jsonl", "--predictor", PREDICTOR]) == EXIT_USAGE
    assert "not found" in capsys.readouterr().err


def test_demo_run_writes_report_and_prediction_rows(tmp_path, capsys):
    report_path, rows_path = tmp_path / "report.json", tmp_path / "rows.jsonl"

    assert _run(tmp_path, "--out", str(report_path), "--predictions-out", str(rows_path)) == EXIT_OK

    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    assert report["dataset"]["n_records"] == len(rows) == 22
    assert report["run"]["predictor"] == PREDICTOR
    assert any(not row["all_correct"] for row in rows)  # the fake predictor injects noise
    assert "overall" in capsys.readouterr().out


def test_regression_gate_trips_against_a_better_baseline(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    assert _run(tmp_path, "--update-baseline") == EXIT_OK
    data = json.loads(baseline.read_text(encoding="utf-8"))
    data["overall"]["call_type"]["accuracy"] = 1.0  # pretend the previous run was perfect
    baseline.write_text(json.dumps(data), encoding="utf-8")

    assert _run(tmp_path, "--fail-on-regression", "0.02") == EXIT_REGRESSION
    assert "call_type_accuracy" in capsys.readouterr().err


def test_regression_gate_passes_against_itself(tmp_path):
    assert _run(tmp_path, "--update-baseline") == EXIT_OK
    assert _run(tmp_path, "--fail-on-regression", "0.0") == EXIT_OK
