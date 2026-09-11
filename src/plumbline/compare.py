# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Execution / denotation comparison between two SQL result sets.

This is the Spider/BIRD "execution accuracy" metric: rather than comparing
SQL text, run both the predicted and the gold query and compare the rows
they return. Two queries are equivalent if they *denote* the same answer,
however differently they are written.

Algorithm adapted from the reference implementation in
`taoyds/test-suite-sql-eval` (Zhong, Yu, Klein, "Semantic Evaluation for
Text-to-SQL with Distilled Test Suites", EMNLP 2020,
https://aclanthology.org/2020.emnlp-main.29/), the metric used by Spider,
SParC, and CoSQL.

Four design decisions, each tracking a fact about SQL semantics:
- FULL result-set compare, not a first-N-rows sample. Sampling produces
  false MISMATCH when both queries return many rows in different orders.
- "Unorder within row" gives column-order independence cheaply: sort each
  row's values before tupling, instead of enumerating column permutations.
- ORDER BY in the gold SQL switches the compare from multiset (bag) to
  positional. This matches SQL semantics: an unordered query is a set, an
  ordered query is a sequence.
- Float values are rounded to 4 decimals — aggregates like AVG/STDDEV
  produce float jitter across engines and across runs of the same engine.
- NULL is a sentinel string, not Python None, so rows sort and hash cleanly.

What this deliberately does NOT do:
- Type coercion: 1 and "1" compare as different (Spider matches this).
- Column-name matching: names are ignored; only values are compared.
- Test-suite accuracy (perturbed DB copies, EMNLP 2020): single-database
  denotation match has a ~2.5% average / ~8.1% worst-case false-negative
  rate. True test suites need many DB copies and are out of scope here;
  the optional LLM judge is the cheap partial mitigation.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

# Sentinels so NULL / NaN can sort and hash alongside normalized strings.
# Real cell strings will not collide with the \x00 prefix.
_NULL_SENTINEL = "\x00NULL"
_NAN_SENTINEL = "\x00NaN"


def normalize_value(v: int | float | str | bool | None) -> str:
    """Canonicalize one cell so equal-meaning values compare equal.

    Why a single canonical *string* form: Counter() needs hashable keys and
    one representation per value, so 5, 5.0, and "5" must collapse together
    while staying distinct from "five". Floats round to 4 decimals (Spider's
    tolerance for aggregate jitter); strings are stripped and lowercased.
    """
    if v is None:
        return _NULL_SENTINEL
    # bool is an int subclass in Python, so check it before the float branch
    # or True would normalize to "1".
    if isinstance(v, bool):
        return "true" if v else "false"
    try:
        f = float(v)
        if math.isnan(f):
            return _NAN_SENTINEL
        rounded = round(f, 4)
        # Collapse whole floats to their int form so 5.0 and 5 match.
        if rounded == int(rounded):
            return str(int(rounded))
        return f"{rounded}"
    except (ValueError, TypeError):
        pass
    return str(v).strip().lower()


def row_to_tuple(row: dict, unorder_columns: bool = True) -> tuple:
    """Convert a row dict to a canonical, hashable tuple of cell strings.

    With unorder_columns=True the values are sorted before tupling — the
    column-order independence trick. Two rows holding the same data in a
    different column arrangement collapse to the same tuple, at O(c log c)
    instead of enumerating O(c!) permutations.
    """
    vals = [normalize_value(v) for v in row.values()]
    if unorder_columns:
        vals = sorted(vals)
    return tuple(vals)


# Conservative ORDER BY detection: a case-insensitive text match that also
# fires on ORDER BY inside CTEs / subqueries. Intentional — any ORDER BY in
# the query tree means order mattered somewhere, so compare positionally.
_ORDER_BY_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def has_order_by(sql: str) -> bool:
    """True if the SQL contains an ORDER BY clause anywhere in its text."""
    return bool(_ORDER_BY_RE.search(sql or ""))


def compare(
    expected_rows: list[dict],
    agent_rows: list[dict],
    gold_sql: str,
) -> str:
    """Compare two result sets and return an execution-match verdict.

    Args:
        expected_rows: full result set from the gold SQL (list of row dicts).
        agent_rows: full result set from the predicted SQL.
        gold_sql: the gold SQL text, used only to detect ORDER BY.

    Returns one of:
        MATCH       — equivalent after canonicalization.
        MISMATCH    — different row sets / sequences.
        EMPTY_BOTH  — both returned zero rows (no signal; excluded from rates).
        EMPTY_ONE   — exactly one side empty.
    """
    if not expected_rows and not agent_rows:
        return "EMPTY_BOTH"
    if not expected_rows or not agent_rows:
        return "EMPTY_ONE"

    # A column-count difference is not auto-fail: an extra harmless column in
    # the prediction just yields tuples with no twin in the gold, so equality
    # decides it. This mirrors the reference exec_eval, which relies on
    # post-normalization tuple equality rather than checking arity up front.
    exp_tuples = [row_to_tuple(r, unorder_columns=True) for r in expected_rows]
    ag_tuples = [row_to_tuple(r, unorder_columns=True) for r in agent_rows]

    if has_order_by(gold_sql):
        # Gold cared about order → positional compare. Different row counts
        # are different sequences, so this also catches a missed LIMIT or an
        # extra row automatically.
        return "MATCH" if exp_tuples == ag_tuples else "MISMATCH"

    # No ORDER BY → bag/multiset semantics: same rows with the same
    # multiplicities, in any order.
    if Counter(exp_tuples) == Counter(ag_tuples):
        return "MATCH"
    return "MISMATCH"


def is_subset(small: Iterable[dict], big: Iterable[dict]) -> bool:
    """True if every row in `small` appears in `big` under multiset semantics.

    Uses the same normalization as compare(). A diagnosis helper (e.g. "did
    the prediction return a subset of the gold rows?"), not part of the verdict.
    """
    sc = Counter(row_to_tuple(r) for r in small)
    bc = Counter(row_to_tuple(r) for r in big)
    for k, v in sc.items():
        if bc[k] < v:
            return False
    return True
