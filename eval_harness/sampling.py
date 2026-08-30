"""Stratified sampling and train/holdout splitting for the eval dataset.

Pure functions over opaque items + accessor callables — no DB, and the only
randomness is a seeded ``random.Random`` — so results are deterministic and
unit-testable. The stratum is ``(company_id, template_id, call_type_key,
duration_bucket)``; ``industry`` is an optional human-labelled field and is
intentionally NOT a sampling axis, because it is rarely available as structured
data.

Minimums win over the target: if the per-company / per-template minimums select
more than ``target_total``, all of them are kept (the caller logs the overage).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import TypeVar

T = TypeVar("T")

DEFAULT_TARGET_TOTAL = 450
DEFAULT_MIN_PER_COMPANY = 30
DEFAULT_MIN_PER_TEMPLATE = 30
DEFAULT_HOLDOUT_FRACTION = 0.3
DEFAULT_SEED = 1337


def _group_indices(items: Sequence[T], key_of: Callable[[T], Hashable]) -> dict[Hashable, list[int]]:
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[key_of(item)].append(index)
    return groups


def _take_minimum(groups: dict[Hashable, list[int]], *, minimum: int, rng: random.Random, selected: set[int]) -> None:
    """Top each group up to ``minimum`` selected indices (counting prior picks)."""
    for _key, indices in sorted(groups.items(), key=lambda kv: repr(kv[0])):
        deficit = minimum - sum(1 for index in indices if index in selected)
        if deficit <= 0:
            continue
        pool = [index for index in indices if index not in selected]
        rng.shuffle(pool)
        selected.update(pool[:deficit])


def _fill_to_target(strata: dict[Hashable, list[int]], *, target: int, rng: random.Random, selected: set[int]) -> None:
    """Round-robin across strata until ``target`` is reached or pools run dry."""
    pools: dict[Hashable, list[int]] = {}
    for key, indices in strata.items():
        pool = [index for index in indices if index not in selected]
        rng.shuffle(pool)
        pools[key] = pool
    keys = sorted(pools, key=repr)
    progress = True
    while len(selected) < target and progress:
        progress = False
        for key in keys:
            if len(selected) >= target:
                break
            if pools[key]:
                selected.add(pools[key].pop())
                progress = True


def stratified_sample(
    items: Sequence[T],
    *,
    stratum_of: Callable[[T], Hashable],
    company_of: Callable[[T], Hashable],
    template_of: Callable[[T], Hashable],
    target_total: int = DEFAULT_TARGET_TOTAL,
    min_per_company: int = DEFAULT_MIN_PER_COMPANY,
    min_per_template: int = DEFAULT_MIN_PER_TEMPLATE,
    seed: int = DEFAULT_SEED,
) -> list[T]:
    """Select a stratified subset honouring per-company / per-template minimums.

    Returns the selected items in their original order. Deterministic for a
    given ``seed`` and input order.
    """
    if not items:
        return []
    rng = random.Random(seed)
    selected: set[int] = set()
    _take_minimum(_group_indices(items, company_of), minimum=min_per_company, rng=rng, selected=selected)
    _take_minimum(_group_indices(items, template_of), minimum=min_per_template, rng=rng, selected=selected)
    _fill_to_target(_group_indices(items, stratum_of), target=target_total, rng=rng, selected=selected)
    return [items[index] for index in sorted(selected)]


def stratified_split(
    items: Sequence[T],
    *,
    stratum_of: Callable[[T], Hashable],
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[list[T], list[T]]:
    """Split into (train, holdout), assigning ~``holdout_fraction`` of EACH stratum.

    Every stratum with >= 2 items yields at least one record to each side, so the
    holdout mirrors the training stratification. A single-item stratum goes to
    train unless ``holdout_fraction`` rounds it into the holdout.
    """
    if not 0.0 <= holdout_fraction <= 1.0:
        raise ValueError(f"holdout_fraction must be within [0, 1], got {holdout_fraction}")
    by_stratum: dict[Hashable, list[T]] = defaultdict(list)
    for item in items:
        by_stratum[stratum_of(item)].append(item)
    rng = random.Random(seed)
    train: list[T] = []
    holdout: list[T] = []
    for stratum in sorted(by_stratum, key=repr):
        group = list(by_stratum[stratum])
        rng.shuffle(group)
        count = round(len(group) * holdout_fraction)
        if len(group) >= 2 and 0.0 < holdout_fraction < 1.0:
            count = max(1, min(count, len(group) - 1))
        holdout.extend(group[:count])
        train.extend(group[count:])
    return train, holdout
