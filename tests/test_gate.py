# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Behavior tests for the gold-set gate.

Real-query defects run against a temp SQLite DB; UNSTABLE uses a stub executor
that returns different rows each call, so the non-determinism is deterministic
in the test (no flaky RANDOM()).
"""
import sqlite3

from plumbline.executor import ExecResult, SQLiteExecutor
from plumbline.gate import (
    DEGENERATE,
    EXEC_FAIL,
    NON_SELECT,
    PLACEHOLDER_COMMENT,
    TRUNCATED,
    UNSTABLE,
    ZERO_ROWS,
    gate,
    gate_one,
    summarize,
)


def _seed(tmp_path):
    """Create a 2-row emp table in a temp DB and return its path."""
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE emp(name TEXT, age INT)")
    con.executemany("INSERT INTO emp VALUES (?, ?)", [("ann", 30), ("bob", 40)])
    con.commit()
    con.close()
    return str(p)


def test_a_good_gold_passes_clean(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    f = gate_one("q1", "SELECT name FROM emp", ex)
    assert f.ok is True
    assert f.defects == ()


def test_non_select_gold_is_flagged(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    f = gate_one("q1", "DELETE FROM emp", ex)
    assert f.defects == (NON_SELECT,)


def test_empty_gold_is_a_placeholder(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    # Only a comment, nothing to run.
    f = gate_one("q1", "-- TODO: write this query", ex)
    assert f.defects == (PLACEHOLDER_COMMENT,)


def test_placeholder_marker_is_flagged_even_with_real_sql(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    f = gate_one("q1", "SELECT name FROM emp -- FIXME wrong table?", ex)
    assert PLACEHOLDER_COMMENT in f.defects


def test_gold_that_errors_is_exec_fail(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    f = gate_one("q1", "SELECT nope FROM emp", ex)
    assert f.defects == (EXEC_FAIL,)
    assert "nope" in f.detail or "no such column" in f.detail.lower()


def test_gold_returning_no_rows_is_zero_rows(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    f = gate_one("q1", "SELECT name FROM emp WHERE age > 999", ex)
    assert f.defects == (ZERO_ROWS,)


def test_empty_aggregate_is_degenerate_not_zero_rows(tmp_path):
    # SUM over an empty filter returns ONE row holding NULL, so ZERO_ROWS would
    # miss it — DEGENERATE is the check that catches a non-answer aggregate.
    ex = SQLiteExecutor(_seed(tmp_path))
    f = gate_one("q1", "SELECT SUM(age) FROM emp WHERE age > 999", ex)
    assert f.defects == (DEGENERATE,)


def test_count_zero_is_degenerate(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    f = gate_one("q1", "SELECT COUNT(*) FROM emp WHERE age > 999", ex)
    assert f.defects == (DEGENERATE,)


def test_row_cap_hit_is_truncated(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path), row_cap=1)
    f = gate_one("q1", "SELECT name FROM emp", ex)
    assert TRUNCATED in f.defects


class _FlakyExecutor:
    """Returns a different single row on each call — a non-deterministic gold."""

    def __init__(self):
        self.calls = 0

    def execute(self, sql):
        self.calls += 1
        return ExecResult(rows=[{"n": self.calls}])


def test_stability_check_flags_a_nondeterministic_gold():
    ex = _FlakyExecutor()
    f = gate_one("q1", "SELECT n FROM t", ex, check_stability=True)
    assert UNSTABLE in f.defects
    assert ex.calls == 2  # ran twice to compare


def test_stability_check_is_off_by_default():
    ex = _FlakyExecutor()
    f = gate_one("q1", "SELECT n FROM t", ex)
    assert UNSTABLE not in f.defects
    assert ex.calls == 1  # ran once, no second run


def test_gate_runs_a_whole_list_and_summarize_counts_defects(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    golds = [
        ("ok", "SELECT name FROM emp"),
        ("bad1", "DELETE FROM emp"),
        ("bad2", "SELECT nope FROM emp"),
    ]
    findings = gate(golds, ex)
    assert [f.ok for f in findings] == [True, False, False]
    assert summarize(findings) == {NON_SELECT: 1, EXEC_FAIL: 1}
