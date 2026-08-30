"""JSONL persistence for eval dataset records.

One :class:`~eval_harness.schema.EvalRecord` per line. Writes are atomic
(temp file + ``os.replace``) so an interrupted labelling save never truncates
the dataset.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Iterator

from eval_harness.schema import EvalRecord


def write_records(path: str, records: Iterable[EvalRecord]) -> int:
    """Write ``records`` to ``path`` as JSONL atomically; return the count written."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    count = 0
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json())
                handle.write("\n")
                count += 1
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return count


def iter_records(path: str) -> Iterator[EvalRecord]:
    """Yield records from a JSONL file, skipping blank lines."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield EvalRecord.model_validate_json(stripped)


def read_records(path: str) -> list[EvalRecord]:
    """Read every record of a JSONL file into memory."""
    return list(iter_records(path))
