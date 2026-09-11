# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Routing accuracy: did the system pick the right table for the question?

In a warehouse with many tables, a text-to-SQL pipeline usually has a first
stage that routes the question to one table (`dataset.table`), and a second
stage that writes SQL against it. A routing miss guarantees a generation miss,
so routing is scored on its own — the report should say WHICH stage failed.

`evaluate_routing` grades predicted tables against gold tables and returns a
RoutingReport with two rates, each with a Wilson confidence interval:

- dataset accuracy — the predicted table is in the right dataset (the first
  dot-separated part matches). Reported because most routing misses land in
  the right dataset but the wrong table; one exact-match number hides that.
- table accuracy   — the full `dataset.table` string matches exactly.

Unlike the SQL harness, nothing is ever excluded from the denominator: an
empty or unparseable prediction is a real wrong answer, not missing data. The
harness's exclusions (GOLD_ERROR, EMPTY_BOTH) exist because a broken gold
can't grade anything — here a broken gold raises instead, since a routing gold
is a single table name and cheap to fix.

Optional catalog snapping: a model that generates the table name freely can
blend two real names into one that does not exist. When a catalog (the set of
valid `dataset.table` names) is given, a near-miss prediction is snapped onto
the closest valid name before scoring, and the report carries `n_snapped` so
the correction is never invisible. Two runs are only comparable when they
agree on whether a catalog was used.
"""
from __future__ import annotations

from dataclasses import dataclass
import difflib

from .metrics import Proportion, proportion

# A routing row: (id, question, gold_table, predicted_table).
RoutingRow = tuple[str, str, str, str]

# Snap threshold for difflib similarity (0..1). 0.8 is tight enough that a
# blend of two real names snaps to the nearer one, while a table that simply
# is not in the catalog stays unmatched instead of snapping to a random entry.
SNAP_CUTOFF = 0.8


@dataclass(frozen=True)
class RoutingResult:
    """The outcome of scoring one routing prediction.

    `pred_table` is the normalized (and possibly snapped) name that was
    actually scored; `raw_prediction` keeps the input as given, so a report
    can show what the snap or normalization changed.
    """

    id: str
    question: str
    gold_table: str
    pred_table: str
    raw_prediction: str
    snapped: bool
    dataset_correct: bool
    table_correct: bool


@dataclass(frozen=True)
class RoutingReport:
    """All per-question results plus the two headline rates with CIs.

    There is no n_scored/n_excluded pair here on purpose: routing has no
    exclusions, so the denominator of both rates is always `n_total`.
    `n_snapped` says how many predictions the catalog snap corrected — a
    reader comparing two runs needs it to know the rates are comparable.
    """

    results: list[RoutingResult]
    n_total: int
    n_snapped: int
    dataset: Proportion
    table: Proportion

    def correctness_vector(self, dataset: bool = False) -> list[bool]:
        """Per-question correctness in input order, for McNemar comparisons.

        Run the same questions in the same order through two routers and the
        two vectors line up by position, as metrics.mcnemar_from_vectors
        requires. Pass dataset=True to compare on the dataset tier instead of
        exact table.
        """
        field = "dataset_correct" if dataset else "table_correct"
        return [getattr(r, field) for r in self.results]


def normalize_table_name(name: str) -> str:
    """Reduce a fully-qualified table name to the canonical `dataset.table`.

    Model output arrives with decoration that should not count as a miss:
    surrounding whitespace, quotes or backticks, a `bigquery:` prefix, or a
    3-part `project.dataset.table`. Everything past that is left alone —
    comparison stays case-sensitive because warehouse table names are.
    """
    name = name.strip().strip("'\"`")
    if name.startswith("bigquery:"):
        name = name[len("bigquery:"):]
    parts = name.split(".")
    if len(parts) == 3:
        return f"{parts[1]}.{parts[2]}"
    return name


def snap_to_catalog(name: str, catalog: set[str]) -> tuple[str, bool]:
    """Snap a near-miss table name onto the closest catalog entry.

    Returns (name, snapped). An empty name or an exact catalog member passes
    through untouched. The catalog is sorted before matching so a similarity
    tie always resolves the same way across runs.
    """
    if not name or name in catalog:
        return name, False
    close = difflib.get_close_matches(name, sorted(catalog), n=1, cutoff=SNAP_CUTOFF)
    return (close[0], True) if close else (name, False)


def _score_one(
    row: RoutingRow, catalog: set[str] | None
) -> RoutingResult:
    """Normalize, optionally snap, then score both tiers for one row."""
    id_, question, raw_gold, raw_pred = row

    gold_table = normalize_table_name(raw_gold)
    if not gold_table:
        raise ValueError(f"row {id_!r}: empty gold_table — fix the answer key")

    pred_table = normalize_table_name(raw_pred)
    snapped = False
    if catalog is not None:
        pred_table, snapped = snap_to_catalog(pred_table, catalog)

    gold_dataset = gold_table.split(".")[0]
    pred_dataset = pred_table.split(".")[0] if pred_table else ""

    return RoutingResult(
        id=id_,
        question=question,
        gold_table=gold_table,
        pred_table=pred_table,
        raw_prediction=raw_pred,
        snapped=snapped,
        dataset_correct=bool(pred_dataset) and pred_dataset == gold_dataset,
        table_correct=pred_table == gold_table,
    )


def evaluate_routing(
    rows: list[RoutingRow],
    catalog: set[str] | None = None,
    confidence: float = 0.95,
) -> RoutingReport:
    """Score every routing prediction and return a RoutingReport.

    Args:
        rows: list of (id, question, gold_table, predicted_table).
        catalog: optional set of valid `dataset.table` names. When given,
            near-miss predictions snap to the closest entry before scoring
            (and the snap count is reported). When None, no snapping.
        confidence: confidence level for both Wilson intervals (default 0.95).

    Raises:
        ValueError: a row's gold_table is empty after normalization. A broken
            answer key fails loudly instead of skewing the rates.
    """
    results = [_score_one(row, catalog) for row in rows]

    n_total = len(results)
    k_dataset = sum(1 for r in results if r.dataset_correct)
    k_table = sum(1 for r in results if r.table_correct)

    return RoutingReport(
        results=results,
        n_total=n_total,
        n_snapped=sum(1 for r in results if r.snapped),
        dataset=proportion(k_dataset, n_total, confidence),
        table=proportion(k_table, n_total, confidence),
    )
