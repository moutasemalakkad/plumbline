# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Behavior tests for routing accuracy.

Pure functions over in-memory rows — no DB, no cloud. Covers the name
normalization, the two scoring tiers, the miss-not-exclusion rule for empty
predictions, catalog snapping on and off, and the report math against the
same pinned Wilson oracle test_metrics.py uses.
"""
import pytest

from plumbline.metrics import mcnemar_from_vectors
from plumbline.routing import (
    evaluate_routing,
    normalize_table_name,
    snap_to_catalog,
)


def _row(pred_table, gold_table="sales.orders", id_="q1"):
    """One routing row with defaults; tests only vary what they assert on."""
    return (id_, "which table?", gold_table, pred_table)


# --- normalization ---


def test_normalize_strips_whitespace_and_quotes():
    assert normalize_table_name("  'sales.orders'  ") == "sales.orders"


def test_normalize_strips_backticks():
    assert normalize_table_name("`sales.orders`") == "sales.orders"


def test_normalize_strips_bigquery_prefix():
    assert normalize_table_name("bigquery:sales.orders") == "sales.orders"


def test_normalize_collapses_three_part_name_to_dataset_table():
    assert normalize_table_name("analytics-prod.sales.orders") == "sales.orders"


def test_normalize_leaves_two_part_name_unchanged():
    assert normalize_table_name("sales.orders") == "sales.orders"


# --- scoring tiers ---


def test_exact_match_scores_both_tiers():
    report = evaluate_routing([_row("sales.orders")])
    result = report.results[0]
    assert result.dataset_correct and result.table_correct


def test_right_dataset_wrong_table_scores_dataset_tier_only():
    report = evaluate_routing([_row("sales.customers")])
    result = report.results[0]
    assert result.dataset_correct and not result.table_correct


def test_empty_prediction_is_a_miss_not_an_exclusion():
    report = evaluate_routing([_row("")])
    result = report.results[0]
    assert not result.dataset_correct and not result.table_correct
    # The empty row stays in the denominator — n never shrinks.
    assert report.n_total == 1
    assert report.table.n == 1 and report.dataset.n == 1


def test_gold_is_normalized_the_same_as_the_prediction():
    # A 3-part, quoted gold still matches a clean 2-part prediction.
    report = evaluate_routing(
        [("q1", "which table?", "'analytics-prod.sales.orders'", "sales.orders")]
    )
    assert report.results[0].table_correct


def test_empty_gold_raises_value_error():
    with pytest.raises(ValueError, match="empty gold_table"):
        evaluate_routing([_row("sales.orders", gold_table="  ")])


# --- catalog snapping ---

_CATALOG = {"sales.orders", "sales.customers", "inventory.stock_levels"}


def test_exact_catalog_entry_passes_through_unsnapped():
    assert snap_to_catalog("sales.orders", _CATALOG) == ("sales.orders", False)


def test_near_miss_snaps_to_catalog_and_is_counted():
    # A dropped plural — the classic free-generation typo.
    report = evaluate_routing(
        [_row("inventory.stock_level", gold_table="inventory.stock_levels")],
        catalog=_CATALOG,
    )
    result = report.results[0]
    assert result.snapped and result.table_correct
    assert result.pred_table == "inventory.stock_levels"
    assert report.n_snapped == 1


def test_unmatchable_prediction_stays_unchanged_and_wrong():
    report = evaluate_routing([_row("finance.invoices")], catalog=_CATALOG)
    result = report.results[0]
    assert not result.snapped and not result.table_correct
    assert result.pred_table == "finance.invoices"


def test_empty_prediction_never_snaps():
    assert snap_to_catalog("", _CATALOG) == ("", False)


def test_no_catalog_means_no_snapping():
    # Same near-miss as above, but without a catalog it stays a plain miss.
    report = evaluate_routing(
        [_row("inventory.stock_level", gold_table="inventory.stock_levels")]
    )
    assert report.n_snapped == 0
    assert not report.results[0].table_correct


# --- report math and comparison ---


def test_report_intervals_match_the_published_wilson_oracle():
    # 19 exact hits out of 35 — same oracle as test_metrics.py: [0.382, 0.695].
    rows = [
        _row("sales.orders" if i < 19 else "sales.customers", id_=f"q{i}")
        for i in range(35)
    ]
    report = evaluate_routing(rows)
    assert report.table.k == 19 and report.table.n == 35
    assert round(report.table.low, 3) == 0.382
    assert round(report.table.high, 3) == 0.695


def test_correctness_vector_feeds_mcnemar():
    # Router A gets q1 right and q2 wrong; router B the reverse -> b=1, c=1.
    a = evaluate_routing([_row("sales.orders", id_="q1"), _row("hr.employees", id_="q2")])
    b = evaluate_routing([_row("hr.employees", id_="q1"), _row("sales.orders", id_="q2")])
    result = mcnemar_from_vectors(a.correctness_vector(), b.correctness_vector())
    assert (result.b, result.c) == (1, 1)
    assert result.p_value == 1.0
