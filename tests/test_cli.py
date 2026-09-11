# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Behavior tests for the command-line interface.

Each test writes a temp SQLite DB and CSV(s), runs main() with argv, and checks
the printed report and exit code — no cloud, the judge stays off.
"""
import sqlite3

from plumbline.cli import format_gate_report, main
from plumbline.gate import GateFinding


def _seed_db(tmp_path):
    """Create a 2-row emp table in a temp DB and return its path."""
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE emp(name TEXT, age INT)")
    con.executemany("INSERT INTO emp VALUES (?, ?)", [("ann", 30), ("bob", 40)])
    con.commit()
    con.close()
    return str(p)


def _write_csv(path, header, rows):
    """Write a CSV with a header line and quoted rows."""
    import csv

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def test_score_prints_rates_and_exits_zero(tmp_path, capsys):
    db = _seed_db(tmp_path)
    preds = _write_csv(
        tmp_path / "p.csv",
        ["id", "question", "gold_sql", "pred_sql"],
        [
            ("q1", "names?", "SELECT name FROM emp", "SELECT name FROM emp"),       # right
            ("q2", "old?", "SELECT name FROM emp WHERE age > 35", "SELECT name FROM emp"),  # wrong
        ],
    )
    code = main(["score", "--db", db, "--predictions", preds])
    out = capsys.readouterr().out
    assert code == 0
    assert "execution accuracy: 0.500" in out
    assert "1/2" in out


def test_gate_reports_clean_set_and_exits_zero(tmp_path, capsys):
    db = _seed_db(tmp_path)
    golds = _write_csv(
        tmp_path / "g.csv",
        ["id", "gold_sql"],
        [("q1", "SELECT name FROM emp")],
    )
    code = main(["gate", "--db", db, "--golds", golds])
    out = capsys.readouterr().out
    assert code == 0
    assert "1 clean, 0 flagged" in out


def test_gate_report_aligns_the_defect_column_across_uneven_ids():
    # Different-length ids must not stagger the defect column.
    findings = [
        GateFinding("a", ("ZERO_ROWS",), "returns no rows"),
        GateFinding("much_longer_id", ("EXEC_FAIL",), "boom"),
    ]
    out = format_gate_report(findings)
    flagged = [line for line in out.splitlines() if line.startswith("  ")]
    starts = {line.index(line.split()[1]) for line in flagged}  # column of the defect token
    assert len(starts) == 1


def test_gate_exits_nonzero_when_a_gold_is_flagged(tmp_path, capsys):
    db = _seed_db(tmp_path)
    golds = _write_csv(
        tmp_path / "g.csv",
        ["id", "gold_sql"],
        [
            ("good", "SELECT name FROM emp"),
            ("bad", "SELECT nope FROM emp"),  # errors -> EXEC_FAIL
        ],
    )
    code = main(["gate", "--db", db, "--golds", golds])
    out = capsys.readouterr().out
    assert code == 1  # so a CI step fails on a defective answer key
    assert "EXEC_FAIL" in out
    assert "bad" in out


def test_score_routing_prints_both_rates_and_exits_zero(tmp_path, capsys):
    preds = _write_csv(
        tmp_path / "r.csv",
        ["id", "question", "gold_table", "pred_table"],
        [
            ("r1", "orders?", "sales.orders", "sales.orders"),      # exact
            ("r2", "returns?", "sales.returns", "sales.orders"),    # dataset only
        ],
    )
    code = main(["score-routing", "--predictions", preds])
    out = capsys.readouterr().out
    assert code == 0
    assert "dataset accuracy: 1.000" in out
    assert "table accuracy:   0.500" in out
    assert "dataset only" in out
    assert "snapped" not in out  # no catalog -> no snap line at all


def test_score_routing_with_catalog_reports_snap_count(tmp_path, capsys):
    preds = _write_csv(
        tmp_path / "r.csv",
        ["id", "question", "gold_table", "pred_table"],
        [
            ("r1", "stock?", "inventory.stock_levels", "inventory.stock_level"),
        ],
    )
    catalog = tmp_path / "catalog.txt"
    catalog.write_text("# valid tables\ninventory.stock_levels\nsales.orders\n")
    code = main(
        ["score-routing", "--predictions", preds, "--catalog", str(catalog)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "snapped 1 of 1 predictions to the catalog" in out
    assert "table accuracy:   1.000" in out  # the snap upgraded the miss
    assert "inventory.stock_level -> inventory.stock_levels" in out
