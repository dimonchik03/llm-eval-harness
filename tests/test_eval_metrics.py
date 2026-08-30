"""Pure-function tests for the eval-harness metrics.

No DB and no network — every value here is hand-computed, so these run on a
bare host. The runner tests live in ``test_eval_harness.py``.
"""

from pytest import approx

from eval_harness import metrics


# --------------------------------------------------------------------------- #
# accuracy
# --------------------------------------------------------------------------- #
def test_accuracy_ignores_null_truth_rows():
    pairs = [("a", "a"), ("b", "b"), ("a", "b"), (None, "a")]
    result = metrics.accuracy(pairs)
    assert result["accuracy"] == approx(2 / 3)
    assert (result["correct"], result["support"]) == (2, 3)


def test_accuracy_normalizes_case_and_whitespace():
    assert metrics.accuracy([("Hot", " hot ")])["accuracy"] == 1.0


def test_accuracy_empty_is_none():
    assert metrics.accuracy([])["accuracy"] is None
    assert metrics.accuracy([])["ci95"] is None


# --------------------------------------------------------------------------- #
# wilson_interval
# --------------------------------------------------------------------------- #
def test_wilson_interval_known_value():
    low, high = metrics.wilson_interval(8, 10)
    assert low == approx(0.4902, abs=1e-4)
    assert high == approx(0.9433, abs=1e-4)


def test_wilson_interval_stays_in_bounds_at_extremes():
    assert metrics.wilson_interval(0, 5)[0] == 0.0
    assert metrics.wilson_interval(5, 5)[1] == 1.0
    assert 0.5 < metrics.wilson_interval(5, 5)[0] < 1.0  # 5/5 is not proof of 100%
    assert metrics.wilson_interval(0, 0) is None


# --------------------------------------------------------------------------- #
# per_class_prf
# --------------------------------------------------------------------------- #
def test_per_class_prf_counts_and_aggregates():
    result = metrics.per_class_prf([("a", "a"), ("a", "b"), ("b", "b")])
    per = result["per_class"]
    assert per["a"]["precision"] == approx(1.0)
    assert per["a"]["recall"] == approx(0.5)
    assert per["a"]["f1"] == approx(2 / 3)
    assert per["b"]["precision"] == approx(0.5)
    assert per["b"]["recall"] == approx(1.0)
    assert result["macro"]["precision"] == approx(0.75)
    assert result["micro"]["f1"] == approx(2 / 3)


# --------------------------------------------------------------------------- #
# confusion_matrix
# --------------------------------------------------------------------------- #
def test_confusion_matrix_groups_and_marks_absent():
    matrix = metrics.confusion_matrix([("a", "a"), ("a", "b"), (None, "a")])
    assert matrix["a"] == {"a": 1, "b": 1}
    assert matrix["∅"] == {"a": 1}


# --------------------------------------------------------------------------- #
# multilabel_prf (risks)
# --------------------------------------------------------------------------- #
def test_multilabel_prf_per_category_and_micro():
    result = metrics.multilabel_prf([(["x", "y"], ["x"]), (["y"], ["y", "z"])])
    per = result["per_category"]
    assert per["x"]["f1"] == approx(1.0)
    assert per["y"]["precision"] == approx(1.0)
    assert per["y"]["recall"] == approx(0.5)
    assert per["z"]["precision"] == approx(0.0)
    assert per["z"]["recall"] is None  # never present in truth
    assert result["micro"]["precision"] == approx(2 / 3)
    assert result["micro"]["recall"] == approx(2 / 3)


# --------------------------------------------------------------------------- #
# keyed_value_prf (custom fields / extensions)
# --------------------------------------------------------------------------- #
def test_keyed_value_prf_value_mismatch_is_fp_and_fn():
    items = [
        ({"budget": "5000", "timeline": "Q1"}, {"budget": "5000"}),
        ({"budget": "100"}, {"budget": "200"}),
    ]
    result = metrics.keyed_value_prf(items)
    per = result["per_slug"]
    assert per["budget"] == {
        "precision": approx(0.5),
        "recall": approx(0.5),
        "f1": approx(0.5),
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "support": 2,
    }
    assert per["timeline"]["recall"] == approx(0.0)
    assert per["timeline"]["precision"] is None  # never predicted
    assert result["micro"]["precision"] == approx(0.5)
    assert result["micro"]["recall"] == approx(1 / 3)


def test_keyed_value_prf_empty_values_are_absent():
    # An empty-string prediction does not "fire" the slug.
    result = metrics.keyed_value_prf([({"a": "x"}, {"a": ""})])
    assert result["per_slug"]["a"]["tp"] == 0
    assert result["per_slug"]["a"]["fn"] == 1


# --------------------------------------------------------------------------- #
# bucketize
# --------------------------------------------------------------------------- #
def test_bucketize_groups_by_key_and_skips_none():
    records = [("a", 1), ("a", 2), ("b", 3), (None, 4)]
    out = metrics.bucketize(records, key_of=lambda r: r[0], metric_of=lambda group: {"n": len(group)})
    assert out == {"a": {"n": 2}, "b": {"n": 1}}  # None key dropped
