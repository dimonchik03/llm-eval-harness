"""Stratified sampling and splitting — minimums, target, determinism."""

from collections import Counter

import pytest

from eval_harness.sampling import stratified_sample, stratified_split


def _calls():
    # Company 1 is large, company 2 is small: a random sample would drown company 2.
    calls = [{"id": i, "company": 1, "template": 1, "type": "sales"} for i in range(200)]
    calls += [{"id": 1000 + i, "company": 2, "template": 2, "type": "support"} for i in range(12)]
    return calls


def _sample(calls, **kwargs):
    return stratified_sample(
        calls,
        stratum_of=lambda c: (c["company"], c["template"], c["type"]),
        company_of=lambda c: c["company"],
        template_of=lambda c: c["template"],
        **kwargs,
    )


def test_small_company_minimum_is_honoured():
    sample = _sample(_calls(), target_total=40, min_per_company=10, min_per_template=0)
    by_company = Counter(c["company"] for c in sample)
    assert by_company[2] >= 10
    assert len(sample) == 40


def test_minimums_win_over_target():
    sample = _sample(_calls(), target_total=5, min_per_company=10, min_per_template=0)
    assert len(sample) == 20  # 10 per company, even though the target is 5


def test_sample_is_deterministic_and_keeps_input_order():
    calls = _calls()
    first = _sample(calls, target_total=30, seed=7)
    assert first == _sample(calls, target_total=30, seed=7)
    assert [c["id"] for c in first] == sorted(c["id"] for c in first)
    assert first != _sample(calls, target_total=30, seed=8)


def test_empty_input():
    assert _sample([]) == []


def test_split_puts_every_multi_item_stratum_on_both_sides():
    items = [("a", i) for i in range(10)] + [("b", i) for i in range(2)] + [("c", 0)]
    train, holdout = stratified_split(items, stratum_of=lambda x: x[0], holdout_fraction=0.3)
    assert sorted(train + holdout) == sorted(items)
    for stratum in ("a", "b"):
        assert any(x[0] == stratum for x in train)
        assert any(x[0] == stratum for x in holdout)
    assert Counter(x[0] for x in holdout)["a"] == 3


def test_split_is_deterministic():
    items = list(range(50))
    assert stratified_split(items, stratum_of=lambda x: x % 3) == stratified_split(items, stratum_of=lambda x: x % 3)


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        stratified_split([1, 2], stratum_of=lambda x: x, holdout_fraction=1.5)
