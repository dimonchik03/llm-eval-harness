"""Pure metric functions for the eval harness.

No DB, no network, no third-party deps — accuracy (with a Wilson confidence
interval), per-class precision/recall/F1, confusion matrices and per-slug
custom-field scores are computed in plain Python so they are trivially
unit-testable and add nothing to the runtime image. Every report block is also
sliceable per company / template / industry via :func:`bucketize`.

Conventions:
- A truth or prediction value is *absent* when it is ``None`` or normalizes to an
  empty string. Accuracy and single-label metrics only count rows whose **truth** is
  present, so unlabelled records never distort a score.
- Values are compared after :func:`normalize` (``str``-cast, stripped, case-folded) so
  trivial casing/whitespace differences do not register as errors.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

R = TypeVar("R")
Pair = tuple[Any, Any]  # (truth, prediction)
_NULL = "∅"
_Z95 = 1.959963984540054  # two-sided 95% normal quantile


def normalize(value: Any) -> str | None:
    """Normalize a label/prediction for comparison; absent -> ``None``."""
    if value is None:
        return None
    text = str(value).strip().casefold()
    return text or None


def wilson_interval(successes: int, total: int, z: float = _Z95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion; ``None`` when ``total`` is 0.

    Preferred over the normal approximation because eval slices are small: it
    stays inside [0, 1] and is sensible at 0% and 100%.
    """
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Precision / recall / F1 from raw counts; ``None`` where undefined."""
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1: float | None
    if precision is not None and recall is not None and (precision + recall):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if (tp + fp + fn) else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
    }


def _macro(per_class: Mapping[str, Mapping[str, Any]]) -> dict[str, float | None]:
    """Unweighted mean of precision/recall/F1 across classes (ignoring ``None``)."""
    out: dict[str, float | None] = {}
    for key in ("precision", "recall", "f1"):
        values = [m[key] for m in per_class.values() if m[key] is not None]
        out[key] = (sum(values) / len(values)) if values else None
    return out


def _micro(per_class: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled precision/recall/F1 across all classes."""
    return _prf(
        sum(m["tp"] for m in per_class.values()),
        sum(m["fp"] for m in per_class.values()),
        sum(m["fn"] for m in per_class.values()),
    )


def accuracy(pairs: Sequence[Pair]) -> dict[str, Any]:
    """Single-label accuracy over rows whose **truth** is present, with a 95% Wilson CI."""
    considered = [(t, p) for t, p in pairs if normalize(t) is not None]
    correct = sum(1 for t, p in considered if normalize(t) == normalize(p))
    total = len(considered)
    return {
        "accuracy": (correct / total) if total else None,
        "ci95": wilson_interval(correct, total),
        "correct": correct,
        "support": total,
    }


def per_class_prf(pairs: Sequence[Pair]) -> dict[str, Any]:
    """Per-class precision/recall/F1 (+ macro/micro) for single-label classification."""
    labels: set[str] = set()
    for truth, pred in pairs:
        labels.update(label for label in (normalize(truth), normalize(pred)) if label is not None)

    normalized = [(normalize(t), normalize(p)) for t, p in pairs]
    per_class: dict[str, dict[str, Any]] = {}
    for label in sorted(labels):
        tp = sum(1 for t, p in normalized if p == label and t == label)
        fp = sum(1 for t, p in normalized if p == label and t != label)
        fn = sum(1 for t, p in normalized if p != label and t == label)
        per_class[label] = _prf(tp, fp, fn)

    return {"per_class": per_class, "macro": _macro(per_class), "micro": _micro(per_class)}


def confusion_matrix(pairs: Sequence[Pair]) -> dict[str, dict[str, int]]:
    """Nested ``{truth: {prediction: count}}`` map; absent values render as ``∅``."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for truth, pred in pairs:
        matrix[normalize(truth) or _NULL][normalize(pred) or _NULL] += 1
    return {truth: dict(row) for truth, row in matrix.items()}


def multilabel_prf(items: Sequence[tuple[Iterable[str], Iterable[str]]]) -> dict[str, Any]:
    """Per-category present/absent P/R/F1 (+ macro/micro) for multi-label sets (e.g. risks)."""
    normalized: list[tuple[set[str], set[str]]] = []
    categories: set[str] = set()
    for truth, pred in items:
        truth_set = {key for key in (normalize(x) for x in truth) if key is not None}
        pred_set = {key for key in (normalize(x) for x in pred) if key is not None}
        normalized.append((truth_set, pred_set))
        categories |= truth_set | pred_set

    per_category: dict[str, dict[str, Any]] = {}
    for category in sorted(categories):
        tp = sum(1 for truth_set, pred_set in normalized if category in truth_set and category in pred_set)
        fp = sum(1 for truth_set, pred_set in normalized if category not in truth_set and category in pred_set)
        fn = sum(1 for truth_set, pred_set in normalized if category in truth_set and category not in pred_set)
        per_category[category] = _prf(tp, fp, fn)

    return {"per_category": per_category, "macro": _macro(per_category), "micro": _micro(per_category)}


def keyed_value_prf(items: Sequence[tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]]) -> dict[str, Any]:
    """Per-key precision/recall/F1 of exact-value match for slug-keyed dicts.

    Used for both custom fields and template extensions. A key "fires" on a side when
    that side has a present value. A true positive needs both sides present **and** an
    equal normalized value; a value mismatch counts as a false positive (for the
    predicted value) and a false negative (for the missed truth) — the standard
    treatment that keeps the micro score honest.
    """
    normalized: list[tuple[dict[str, str], dict[str, str]]] = []
    slugs: set[str] = set()
    for truth, pred in items:
        truth_map = {k: n for k, v in (truth or {}).items() if (n := normalize(v)) is not None}
        pred_map = {k: n for k, v in (pred or {}).items() if (n := normalize(v)) is not None}
        normalized.append((truth_map, pred_map))
        slugs |= set(truth_map) | set(pred_map)

    per_slug: dict[str, dict[str, Any]] = {}
    for slug in sorted(slugs):
        tp = sum(1 for t, p in normalized if slug in t and slug in p and t[slug] == p[slug])
        fp = sum(1 for t, p in normalized if slug in p and (slug not in t or t[slug] != p[slug]))
        fn = sum(1 for t, p in normalized if slug in t and (slug not in p or t[slug] != p[slug]))
        per_slug[slug] = _prf(tp, fp, fn)

    return {"per_slug": per_slug, "macro": _macro(per_slug), "micro": _micro(per_slug)}


def bucketize(
    records: Sequence[R],
    key_of: Callable[[R], Any],
    metric_of: Callable[[list[R]], dict[str, Any]],
) -> dict[str, Any]:
    """Group ``records`` by ``key_of`` (skipping ``None`` keys) and apply ``metric_of`` per group.

    Keys are stringified so the result is JSON-serializable and ordering is stable.
    """
    groups: dict[Any, list[R]] = defaultdict(list)
    for record in records:
        key = key_of(record)
        if key is None:
            continue
        groups[key].append(record)
    return {str(key): metric_of(group) for key, group in sorted(groups.items(), key=lambda kv: str(kv[0]))}
