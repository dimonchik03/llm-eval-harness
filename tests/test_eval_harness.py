"""Runner tests over the ``predict_fn`` seam — no DB, no network.

Exercises ``run_eval`` end-to-end with synthetic records and a deterministic
in-memory predict_fn, plus the judge with its single LLM seam
(``runner._call_judge_llm``) monkeypatched so no key is needed. Asserts the
metric wiring, per-company bucket isolation, the non-analytical judge
short-circuit, judge integration and its fail-closed behaviour, the
unlabelled-skip, error capture and exclusion from quality metrics, per-record
rows, concurrency, regression detection, and — critically — that the harness
itself writes nothing to disk.
"""

from eval_harness import runner
from eval_harness.runner import (
    FaithfulnessVerdict,
    StagePrediction,
    compare_to_baseline,
    default_faithfulness_judge,
    evaluate,
    find_regressions,
    prediction_rows,
    run_eval,
)
from eval_harness.schema import LEAD_QUALITY_VALUES, CallInput, CallLabel, EvalRecord, StratumKey

LV = LEAD_QUALITY_VALUES[0]  # a real lead_quality enum value (label schema validates it)
LABELLED_AT = "2026-01-15T09:00:00+00:00"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _labelled(**overrides):
    base = dict(
        call_type_key="sales_consultation",
        lead_quality=LV,
        custom_fields={"budget": "5000"},
        extensions={"current_funnel_stage": "warm"},
        risk_category_keys=["price_objection"],
        labelled_at=LABELLED_AT,
    )
    base.update(overrides)
    return CallLabel(**base)


def _record(call_id, *, company_id=1, template_id=1, label=None):
    inp = CallInput(
        call_id=call_id,
        company_id=company_id,
        template_id=template_id,
        transcript="manager: good morning\nclient: hi, I want a quote",
    )
    stratum = StratumKey(company_id=company_id, template_id=template_id)
    return EvalRecord(input=inp, label=label if label is not None else _labelled(), stratum=stratum)


def _echo_predict_fn(counters=None, *, call_type=None, is_analytical=True):
    """A predict_fn that replays each record's label (optionally overriding the call type).

    Synthetic stand-in for a real pipeline adapter; ``counters["predict"]`` tracks
    how many records were actually predicted (the unlabelled-skip assertion).
    """

    def predict(record):
        if counters is not None:
            counters["predict"] += 1
        label = record.label
        prediction = StagePrediction(
            call_id=record.input.call_id,
            company_id=record.input.company_id,
            template_id=record.input.template_id,
            call_type_key=call_type if call_type is not None else label.call_type_key,
            is_analytical=is_analytical,
        )
        if is_analytical is not False:
            prediction.lead_quality = label.lead_quality
            prediction.custom_fields = dict(label.custom_fields)
            prediction.extensions = dict(label.extensions)
            prediction.risk_category_keys = list(label.risk_category_keys)
            prediction.summary = "Client asked for a quote on the premium plan."
        return prediction

    return predict


def _counting_judge(counters, verdict):
    def judge(transcript, narrative):
        counters["judge"] += 1
        return verdict

    return judge


# --------------------------------------------------------------------------- #
# Happy path — every metric resolves
# --------------------------------------------------------------------------- #
def test_run_eval_perfect_match():
    record = _record(1)

    report = run_eval([record], _echo_predict_fn(), dataset_path="x.jsonl")

    assert report["dataset"]["n_records"] == 1
    assert report["dataset"]["n_evaluated"] == 1
    assert report["dataset"]["n_errors"] == 0
    overall = report["overall"]
    assert overall["call_type"]["accuracy"] == 1.0
    assert overall["lead_quality"]["accuracy"] == 1.0
    assert overall["risks"]["micro"]["f1"] == 1.0
    assert overall["custom_fields"]["micro"]["f1"] == 1.0
    assert overall["extensions"]["micro"]["f1"] == 1.0
    assert "1" in report["by_company"]
    assert "1" in report["by_template"]


# --------------------------------------------------------------------------- #
# No writes: the harness only reads the records it was handed
# --------------------------------------------------------------------------- #
def test_run_eval_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # any accidental relative-path write would land here

    report = run_eval([_record(1)], _echo_predict_fn())

    assert report["dataset"]["n_evaluated"] == 1
    assert list(tmp_path.iterdir()) == []  # no report, cache, or dataset files materialize


# --------------------------------------------------------------------------- #
# Per-company bucketing isolates tenants
# --------------------------------------------------------------------------- #
def test_run_eval_buckets_per_company():
    # A's label agrees with the (shared) prediction; B's call type disagrees.
    record_a = _record(1, company_id=1, label=_labelled(call_type_key="sales_consultation"))
    record_b = _record(2, company_id=2, label=_labelled(call_type_key="support_request"))

    report = run_eval([record_a, record_b], _echo_predict_fn(call_type="sales_consultation"))

    assert report["overall"]["call_type"]["accuracy"] == 0.5
    assert report["by_company"]["1"]["call_type"]["accuracy"] == 1.0
    assert report["by_company"]["2"]["call_type"]["accuracy"] == 0.0


# --------------------------------------------------------------------------- #
# Non-analytical short-circuit: no analytical payload, judge never runs
# --------------------------------------------------------------------------- #
def test_non_analytical_skips_judge():
    counters = {"predict": 0, "judge": 0}
    predict_fn = _echo_predict_fn(counters, call_type="spam", is_analytical=False)
    # Label carries only a call type (no main-analysis ground truth).
    record = _record(1, label=CallLabel(call_type_key="spam", labelled_at=LABELLED_AT))
    judge = _counting_judge(counters, FaithfulnessVerdict(faithful=True, score=5, unsupported_claims=[]))

    report = run_eval([record], predict_fn, judge=judge)

    assert counters["judge"] == 0  # the judge was skipped for the non-analytical call
    assert report["overall"]["call_type"]["accuracy"] == 1.0
    assert report["overall"]["lead_quality"]["accuracy"] is None  # no truth rows
    assert "summary_faithfulness" not in report["overall"]


# --------------------------------------------------------------------------- #
# Summary-faithfulness judge (default judge, LLM seam monkeypatched)
# --------------------------------------------------------------------------- #
def test_judge_populates_faithfulness(monkeypatch):
    verdict = FaithfulnessVerdict(faithful=True, score=4, unsupported_claims=[])
    seen = {}

    def fake_judge_llm(transcript, prompt, model):
        seen["transcript"] = transcript
        seen["prompt"] = prompt
        return verdict

    monkeypatch.setattr(runner, "_call_judge_llm", fake_judge_llm)
    record = _record(1)

    report = run_eval([record], _echo_predict_fn(), judge=default_faithfulness_judge)

    faithfulness = report["overall"]["summary_faithfulness"]
    assert faithfulness["support"] == 1
    assert faithfulness["mean_score"] == 4
    assert faithfulness["faithful_rate"] == 1.0
    assert seen["transcript"] == record.input.transcript
    assert "premium plan" in seen["prompt"]  # the narrative is embedded in the judge prompt


# --------------------------------------------------------------------------- #
# Unlabelled records are skipped (no metric value, no wasted LLM call)
# --------------------------------------------------------------------------- #
def test_unlabelled_records_skipped():
    counters = {"predict": 0}
    record = _record(1, label=CallLabel())  # labelled_at is None

    report = run_eval([record], _echo_predict_fn(counters))

    assert report["dataset"]["n_records"] == 0
    assert counters["predict"] == 0


# --------------------------------------------------------------------------- #
# A raising predict_fn is captured as an error, never aborts the run
# --------------------------------------------------------------------------- #
def test_predict_fn_exception_is_captured_as_error():
    def broken_predict(record):
        raise RuntimeError("provider timeout")

    report = run_eval([_record(999)], broken_predict)

    assert report["dataset"]["n_evaluated"] == 0
    assert report["dataset"]["n_errors"] == 1
    assert report["errors"][0] == {"call_id": 999, "error": "RuntimeError: provider timeout"}


def test_errors_are_excluded_from_quality_metrics():
    def flaky(record):
        if record.input.call_id == 2:
            raise TimeoutError("provider timeout")
        return _echo_predict_fn()(record)

    report = run_eval([_record(1), _record(2)], flaky)

    overall = report["overall"]
    assert overall["call_type"]["accuracy"] == 1.0  # the outage is not a wrong answer
    assert overall["call_type"]["support"] == 1
    assert overall["n_errors"] == 1
    assert overall["error_rate"] == 0.5


def test_predict_fn_returning_wrong_type_is_an_error():
    report = run_eval([_record(1)], lambda record: {"call_type_key": "x"})
    assert report["errors"][0]["error"].startswith("TypeError")


# --------------------------------------------------------------------------- #
# The judge fails closed
# --------------------------------------------------------------------------- #
def test_judge_refusal_is_a_judge_error_not_a_faithful_summary(monkeypatch):
    monkeypatch.setattr(runner, "_call_judge_llm", lambda transcript, prompt, model: None)

    report = run_eval([_record(1)], _echo_predict_fn(), judge=default_faithfulness_judge)

    faithfulness = report["overall"]["summary_faithfulness"]
    assert faithfulness["n_judge_errors"] == 1
    assert faithfulness["support"] == 0
    assert faithfulness["faithful_rate"] is None


def test_judge_out_of_range_score_is_rejected():
    class Verdict:
        faithful = True
        score = 9

    report = run_eval([_record(1)], _echo_predict_fn(), judge=lambda transcript, narrative: Verdict())
    assert report["overall"]["summary_faithfulness"]["n_judge_errors"] == 1


def test_verdict_has_no_optimistic_defaults():
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        FaithfulnessVerdict()
    with pytest.raises(pydantic.ValidationError):
        FaithfulnessVerdict(faithful=True, score=0, unsupported_claims=[])


def test_judge_runs_when_is_analytical_is_unset():
    counters = {"predict": 0, "judge": 0}
    judge = _counting_judge(counters, FaithfulnessVerdict(faithful=False, score=2, unsupported_claims=["x"]))

    report = run_eval([_record(1)], _echo_predict_fn(is_analytical=None), judge=judge)

    assert counters["judge"] == 1
    assert report["overall"]["summary_faithfulness"]["faithful_rate"] == 0.0


# --------------------------------------------------------------------------- #
# Per-record rows, concurrency, baselines
# --------------------------------------------------------------------------- #
def test_prediction_rows_show_what_went_wrong():
    record = _record(1, label=_labelled(risk_category_keys=["price_objection", "churn_risk"]))

    def predict(rec):
        prediction = _echo_predict_fn(call_type="support_request")(rec)
        prediction.risk_category_keys = ["price_objection", "delivery_concern"]
        prediction.custom_fields = {"budget": "6000"}
        return prediction

    (row,) = prediction_rows(evaluate([record], predict))

    assert row["all_correct"] is False
    assert row["call_type"] == {"truth": "sales_consultation", "pred": "support_request", "match": False}
    assert row["risks"] == {"missing": ["churn_risk"], "extra": ["delivery_concern"]}
    assert row["custom_fields"] == {"budget": {"truth": "5000", "pred": "6000"}}


def test_prediction_rows_all_correct_on_perfect_match():
    (row,) = prediction_rows(evaluate([_record(1)], _echo_predict_fn()))
    assert row["all_correct"] is True


def test_concurrency_preserves_order_and_results():
    records = [_record(i, company_id=i % 3) for i in range(20)]
    sequential = run_eval(records, _echo_predict_fn(call_type="sales_consultation"))
    parallel = run_eval(records, _echo_predict_fn(call_type="sales_consultation"), concurrency=4)
    assert parallel == sequential


def test_find_regressions_flags_only_drops_beyond_threshold():
    current = {"overall": {"call_type": {"accuracy": 0.80}, "lead_quality": {"accuracy": 0.90}}}
    baseline = {"overall": {"call_type": {"accuracy": 0.90}, "lead_quality": {"accuracy": 0.91}}}
    delta = compare_to_baseline(current, baseline)
    regressions = find_regressions(delta, threshold=0.02)
    assert list(regressions) == ["call_type_accuracy"]
    assert regressions["call_type_accuracy"] < -0.09
