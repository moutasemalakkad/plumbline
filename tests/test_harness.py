# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Behavior tests for the end-to-end harness.

Real SQLite execution drives the comparator paths; the judge path uses
echo_backend so no cloud is needed. Each test pins one rule of how raw outcomes
roll up into the two headline rates.
"""
import sqlite3

from plumbline.backends import echo_backend
from plumbline.harness import GOLD_ERROR, PRED_ERROR, evaluate
from plumbline.executor import SQLiteExecutor
from plumbline.metrics import mcnemar_from_vectors


def _seed(tmp_path):
    """Create a 2-row emp table in a temp DB and return its path."""
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE emp(name TEXT, age INT)")
    con.executemany("INSERT INTO emp VALUES (?, ?)", [("ann", 30), ("bob", 40)])
    con.commit()
    con.close()
    return str(p)


def test_a_correct_prediction_scores_as_match(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    # Each prediction is a 4-tuple: (id, question, gold_sql, pred_sql).
    # Here the predicted SQL adds an ORDER BY the gold lacks, so the rows are
    # the same multiset -> MATCH.
    preds = [("q1", "names?", "SELECT name FROM emp", "SELECT name FROM emp ORDER BY name DESC")]
    report = evaluate(ex, preds)
    assert report.results[0].verdict == "MATCH"
    assert report.execution.point == 1.0


def test_a_wrong_prediction_lowers_the_rate(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    preds = [
        ("q1", "names?", "SELECT name FROM emp", "SELECT name FROM emp"),          # right
        ("q2", "old?", "SELECT name FROM emp WHERE age > 35", "SELECT name FROM emp"),  # wrong
    ]
    report = evaluate(ex, preds)
    assert report.n_scored == 2
    assert report.execution.k == 1
    assert report.execution.point == 0.5


def test_a_prediction_that_fails_to_run_counts_as_wrong(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    preds = [("q1", "names?", "SELECT name FROM emp", "SELECT nope FROM emp")]
    report = evaluate(ex, preds)
    assert report.results[0].verdict == PRED_ERROR
    assert report.results[0].execution_correct is False
    assert report.n_scored == 1  # a failed prediction is still scored (as wrong)


def test_a_broken_gold_is_excluded_from_the_rate(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    preds = [("q1", "names?", "SELECT nope FROM emp", "SELECT name FROM emp")]
    report = evaluate(ex, preds)
    assert report.results[0].verdict == GOLD_ERROR
    assert report.n_excluded == 1
    assert report.n_scored == 0


def test_empty_both_is_excluded_from_the_rate(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    # Both gold and prediction return nothing -> no signal -> excluded.
    preds = [(
        "q1", "nobody?",
        "SELECT name FROM emp WHERE age > 999",
        "SELECT name FROM emp WHERE age > 888",
    )]
    report = evaluate(ex, preds)
    assert report.results[0].verdict == "EMPTY_BOTH"
    assert report.n_scored == 0
    assert report.n_excluded == 1


def test_without_a_judge_adjusted_equals_execution(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    preds = [("q1", "old?", "SELECT name FROM emp WHERE age > 35", "SELECT name FROM emp")]
    report = evaluate(ex, preds)
    assert report.execution.point == report.adjusted.point == 0.0


def test_judge_can_upgrade_a_mismatch_but_only_adjusted_moves(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    # This prediction really mismatches; a judge saying EQUIVALENT upgrades it.
    preds = [("q1", "old?", "SELECT name FROM emp WHERE age > 35", "SELECT name FROM emp")]
    backend = echo_backend("VERDICT: EQUIVALENT\nREASON: same intent")
    report = evaluate(ex, preds, judge_backend=backend)
    r = report.results[0]
    assert r.execution_correct is False        # exact comparator unchanged
    assert r.adjusted_correct is True          # judge upgraded it
    assert report.execution.point == 0.0
    assert report.adjusted.point == 1.0


def test_judge_not_equivalent_leaves_the_row_wrong(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    preds = [("q1", "old?", "SELECT name FROM emp WHERE age > 35", "SELECT name FROM emp")]
    backend = echo_backend("VERDICT: NOT_EQUIVALENT\nREASON: different rows")
    report = evaluate(ex, preds, judge_backend=backend)
    assert report.adjusted.point == 0.0


def test_correctness_vector_drops_excluded_and_feeds_mcnemar(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    # q2's gold is broken -> excluded; the vector covers only q1 and q3.
    preds = [
        ("q1", "a", "SELECT name FROM emp", "SELECT name FROM emp"),                 # right
        ("q2", "b", "SELECT nope FROM emp", "SELECT name FROM emp"),                 # excluded
        ("q3", "c", "SELECT name FROM emp WHERE age > 35", "SELECT name FROM emp"),  # wrong
    ]
    report = evaluate(ex, preds)
    assert report.correctness_vector() == [True, False]

    # Two runs scored on the same ordered predictions align by position.
    run_a = [True, False]
    run_b = [True, True]
    result = mcnemar_from_vectors(run_a, run_b)
    assert result.c == 1  # the second run fixed the one the first got wrong
