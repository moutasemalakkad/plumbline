# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Gold-set gate — audit the answer key before trusting any score.

Every accuracy number is computed against the gold queries, so a broken gold
silently corrupts the metric: an empty result, a query that errors, or a stub
left as a comment all make correct predictions look wrong (or wrong ones look
right). This gate runs each gold through an Executor and flags the defect
classes below, so a human fixes the answer key first.

It judges the GOLD, not the predictions — it is upstream of the harness, run
once when a gold set is built or changed.

Defect classes:
- NON_SELECT          gold is not a read-only SELECT/WITH (can't be trusted/run)
- PLACEHOLDER_COMMENT gold is empty after stripping comments, or carries a
                      TODO/FIXME-style marker — an unfinished stub
- EXEC_FAIL           gold raises when executed
- ZERO_ROWS           gold returns no rows (often a wrong filter or stale data)
- DEGENERATE          gold returns a single trivial cell (NULL or 0). Distinct
                      from ZERO_ROWS because an aggregate over an empty set
                      returns one row, not zero — `SELECT SUM(x) WHERE <never>`
                      yields [{None}], which ZERO_ROWS would miss
- TRUNCATED           gold hit the executor's row cap (only fires if the
                      executor was built with a row_cap)
- UNSTABLE            two runs returned different rows (non-deterministic gold,
                      e.g. ORDER BY without a tiebreaker). Opt-in: it runs each
                      gold twice, so it doubles execution cost
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .compare import compare, normalize_value
from .executor import Executor, is_read_only, strip_comments

NON_SELECT = "NON_SELECT"
PLACEHOLDER_COMMENT = "PLACEHOLDER_COMMENT"
EXEC_FAIL = "EXEC_FAIL"
ZERO_ROWS = "ZERO_ROWS"
DEGENERATE = "DEGENERATE"
TRUNCATED = "TRUNCATED"
UNSTABLE = "UNSTABLE"

_PLACEHOLDER_RE = re.compile(r"\b(TODO|FIXME|PLACEHOLDER|XXX)\b", re.IGNORECASE)

# A single cell holding one of these reads as a non-answer, not real data.
_TRIVIAL_CELLS = {normalize_value(None), normalize_value(0)}

# Two runs disagreeing on rows means the gold is non-deterministic. EMPTY_BOTH
# (consistently empty) and MATCH (consistently equal) are both stable.
_UNSTABLE_VERDICTS = ("MISMATCH", "EMPTY_ONE")


@dataclass(frozen=True)
class GateFinding:
    """Defects found for one gold query; `ok` means it passed clean.

    `defects` is a tuple of the class names above (empty when clean). `detail`
    is a human note joining the reasons, for the report a reviewer reads.
    """

    gold_id: str
    defects: tuple[str, ...]
    detail: str = ""

    @property
    def ok(self) -> bool:
        """True if the gold has no defects."""
        return not self.defects


def _is_degenerate(rows: list[dict]) -> bool:
    """True if rows is a single 1-column cell holding NULL or 0."""
    if len(rows) != 1:
        return False
    cells = list(rows[0].values())
    if len(cells) != 1:
        return False
    return normalize_value(cells[0]) in _TRIVIAL_CELLS


def gate_one(
    gold_id: str,
    gold_sql: str,
    executor: Executor,
    check_stability: bool = False,
) -> GateFinding:
    """Audit one gold query and return its GateFinding.

    Text-only defects (empty stub, non-SELECT) short-circuit before running,
    since a stub can't be executed and a non-SELECT must not be. Execution
    defects are then collected from a single run; check_stability adds a second
    run to catch non-deterministic golds (off by default — it doubles cost).
    """
    defects: list[str] = []
    notes: list[str] = []

    stripped = strip_comments(gold_sql).strip().rstrip(";").strip()
    if not stripped:
        return GateFinding(
            gold_id, (PLACEHOLDER_COMMENT,), "empty after stripping comments"
        )
    if _PLACEHOLDER_RE.search(gold_sql or ""):
        defects.append(PLACEHOLDER_COMMENT)
        notes.append("carries a placeholder marker (TODO/FIXME/...)")

    if not is_read_only(gold_sql):
        # Can't safely run a non-SELECT; report and stop here.
        defects.append(NON_SELECT)
        notes.append("not a read-only SELECT/WITH")
        return GateFinding(gold_id, tuple(defects), "; ".join(notes))

    res = executor.execute(gold_sql)
    if not res.ok:
        defects.append(EXEC_FAIL)
        notes.append(f"execution failed: {res.error}")
        return GateFinding(gold_id, tuple(defects), "; ".join(notes))

    if res.truncated:
        defects.append(TRUNCATED)
        notes.append("hit the executor row cap")

    if not res.rows:
        defects.append(ZERO_ROWS)
        notes.append("returns no rows")
    elif _is_degenerate(res.rows):
        defects.append(DEGENERATE)
        notes.append("single trivial cell (NULL/0) — likely an empty aggregate")

    if check_stability:
        second = executor.execute(gold_sql)
        if second.ok and compare(res.rows, second.rows, gold_sql) in _UNSTABLE_VERDICTS:
            defects.append(UNSTABLE)
            notes.append("two runs returned different rows")

    return GateFinding(gold_id, tuple(defects), "; ".join(notes))


def gate(
    golds: list[tuple[str, str]],
    executor: Executor,
    check_stability: bool = False,
) -> list[GateFinding]:
    """Audit a list of (gold_id, gold_sql) pairs; return one finding per gold."""
    return [
        gate_one(gold_id, gold_sql, executor, check_stability)
        for gold_id, gold_sql in golds
    ]


def summarize(findings: list[GateFinding]) -> dict[str, int]:
    """Count how many golds carry each defect class (clean golds contribute none).

    A gold with two defects is counted under both, so the values can sum to more
    than the number of golds — this is a per-defect tally for the report header,
    not a partition.
    """
    counts: dict[str, int] = {}
    for finding in findings:
        for defect in finding.defects:
            counts[defect] = counts.get(defect, 0) + 1
    return counts
