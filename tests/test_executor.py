# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Behavior tests for the SQLite backend and the read-only guard.

BigQueryExecutor needs credentials, so it's out of the fast path; these cover
the stdlib backend and the safety rail that protects every backend.
"""
import sqlite3

from plumbline.executor import (
    ExecResult,
    SQLiteExecutor,
    is_read_only,
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


def test_read_only_accepts_select_and_with():
    assert is_read_only("SELECT 1") is True
    assert is_read_only("WITH t AS (SELECT 1 AS x) SELECT x FROM t") is True


def test_read_only_rejects_data_modifying_statements():
    assert is_read_only("DELETE FROM emp") is False
    assert is_read_only("DROP TABLE emp") is False
    assert is_read_only("UPDATE emp SET age = 0") is False
    assert is_read_only("INSERT INTO emp VALUES ('x', 1)") is False


def test_read_only_rejects_a_second_statement():
    # The classic "SELECT then sneak a write" gets caught by the semicolon.
    assert is_read_only("SELECT 1; DROP TABLE emp") is False


def test_read_only_ignores_keywords_inside_comments():
    # A DROP in a comment is stripped, so the query is still read-only.
    assert is_read_only("-- cleanup DROP TABLE emp\nSELECT 1") is True


def test_select_returns_rows_as_dicts(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    res = ex.execute("SELECT name, age FROM emp ORDER BY age")
    assert res.ok is True
    assert res.rows == [{"name": "ann", "age": 30}, {"name": "bob", "age": 40}]


def test_zero_rows_is_ok_not_error(tmp_path):
    # "ran, returned nothing" must be distinct from "failed to run".
    ex = SQLiteExecutor(_seed(tmp_path))
    res = ex.execute("SELECT name FROM emp WHERE age > 999")
    assert res.ok is True
    assert res.rows == []


def test_bad_sql_becomes_error_not_exception(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path))
    res = ex.execute("SELECT nope FROM emp")
    assert res.ok is False
    assert "nope" in res.error or "no such column" in res.error.lower()


def test_non_read_only_is_rejected_before_touching_db(tmp_path):
    db = _seed(tmp_path)
    ex = SQLiteExecutor(db)
    res = ex.execute("DELETE FROM emp")
    assert res.ok is False
    assert "read-only" in res.error
    # the table is untouched
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM emp").fetchone()[0] == 2
    con.close()


def test_row_cap_truncates_and_flags(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path), row_cap=1)
    res = ex.execute("SELECT name FROM emp ORDER BY age")
    assert len(res.rows) == 1
    assert res.truncated is True


def test_row_cap_not_hit_is_not_truncated(tmp_path):
    ex = SQLiteExecutor(_seed(tmp_path), row_cap=10)
    res = ex.execute("SELECT name FROM emp")
    assert res.truncated is False


def test_exec_result_ok_reflects_error_field():
    assert ExecResult(rows=[]).ok is True
    assert ExecResult(error="boom").ok is False


def test_cli_requires_exactly_one_backend():
    """--db and --bq-project are mutually exclusive, and one is required.

    Cheap guard on the CLI surface: BigQuery runs cost money and need
    credentials, so the selection logic is what a fast test can cover.
    """
    import pytest

    from plumbline.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):  # neither
        parser.parse_args(["gate", "--golds", "g.csv"])
    with pytest.raises(SystemExit):  # both
        parser.parse_args(["gate", "--golds", "g.csv", "--db", "a.db",
                           "--bq-project", "p"])

    args = parser.parse_args(["gate", "--golds", "g.csv", "--bq-project", "p"])
    assert args.bq_project == "p" and args.db is None
    assert args.max_bytes == 1_000_000_000  # guard on by default


def test_make_executor_picks_sqlite_without_bq_project(tmp_path):
    """No --bq-project means SQLite, and no [bigquery] extra is needed."""
    from plumbline.cli import _build_parser, _make_executor
    from plumbline.executor import SQLiteExecutor

    db = tmp_path / "x.db"
    db.touch()
    args = _build_parser().parse_args(
        ["gate", "--golds", "g.csv", "--db", str(db)]
    )
    assert isinstance(_make_executor(args), SQLiteExecutor)
