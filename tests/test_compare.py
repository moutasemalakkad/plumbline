# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Behavior tests for the execution comparator.

Each test pins one rule of SQL result-set equivalence with literal rows, so
the file doubles as the spec for what MATCH/MISMATCH/EMPTY_* mean.
"""
from plumbline.compare import (
    compare,
    has_order_by,
    is_subset,
    normalize_value,
)

# Gold SQL stubs. compare() only reads the gold SQL to detect ORDER BY, so
# these two strings are all that's needed to drive the order-agnostic vs
# order-sensitive code paths.
NO_ORDER = "SELECT n FROM t"
ORDERED = "SELECT n FROM t ORDER BY n"


def test_identical_rows_match():
    rows = [{"n": 1}, {"n": 2}]
    assert compare(rows, rows, NO_ORDER) == "MATCH"


def test_column_order_does_not_matter():
    # Same data, columns (and column names) arranged differently -> MATCH.
    gold = [{"name": "ann", "age": 30}]
    pred = [{"age": 30, "name": "ann"}]
    assert compare(gold, pred, NO_ORDER) == "MATCH"


def test_row_order_does_not_matter_without_order_by():
    gold = [{"n": 1}, {"n": 2}]
    pred = [{"n": 2}, {"n": 1}]
    assert compare(gold, pred, NO_ORDER) == "MATCH"


def test_row_order_matters_when_gold_has_order_by():
    gold = [{"n": 1}, {"n": 2}]
    pred = [{"n": 2}, {"n": 1}]
    assert compare(gold, pred, ORDERED) == "MISMATCH"


def test_same_order_matches_when_gold_has_order_by():
    rows = [{"n": 1}, {"n": 2}]
    assert compare(rows, rows, ORDERED) == "MATCH"


def test_duplicate_multiplicity_matters():
    # Bag semantics: a duplicated row is not the same as a single row.
    gold = [{"n": 1}, {"n": 1}, {"n": 2}]
    pred = [{"n": 1}, {"n": 2}]
    assert compare(gold, pred, NO_ORDER) == "MISMATCH"


def test_both_empty_is_empty_both():
    assert compare([], [], NO_ORDER) == "EMPTY_BOTH"


def test_one_side_empty_is_empty_one():
    assert compare([{"n": 1}], [], NO_ORDER) == "EMPTY_ONE"


def test_floats_within_four_decimals_match():
    # 3.333341 and 3.333344 both round to 3.3333.
    gold = [{"v": 3.333341}]
    pred = [{"v": 3.333344}]
    assert compare(gold, pred, NO_ORDER) == "MATCH"


def test_floats_beyond_four_decimals_mismatch():
    # 3.33331 -> 3.3333 ; 3.33339 -> 3.3334.
    gold = [{"v": 3.33331}]
    pred = [{"v": 3.33339}]
    assert compare(gold, pred, NO_ORDER) == "MISMATCH"


def test_whole_float_equals_int():
    assert compare([{"v": 5.0}], [{"v": 5}], NO_ORDER) == "MATCH"


def test_strings_are_stripped_and_lowercased():
    assert compare([{"s": "Ann"}], [{"s": " ann "}], NO_ORDER) == "MATCH"


def test_null_equals_null():
    assert compare([{"v": None}], [{"v": None}], NO_ORDER) == "MATCH"


def test_null_is_not_zero():
    assert compare([{"v": None}], [{"v": 0}], NO_ORDER) == "MISMATCH"


def test_bool_normalizes_distinct_from_int():
    # bool is checked before float, so True is "true", not "1".
    assert normalize_value(True) == "true"
    assert normalize_value(1) == "1"


def test_has_order_by_detects_nested_clause():
    assert has_order_by("SELECT n FROM (SELECT n FROM t ORDER BY n) x") is True
    assert has_order_by("SELECT n FROM t") is False


def test_is_subset_true_when_all_rows_present():
    small = [{"n": 1}]
    big = [{"n": 1}, {"n": 2}]
    assert is_subset(small, big) is True


def test_is_subset_false_when_a_row_is_missing():
    assert is_subset([{"n": 3}], [{"n": 1}, {"n": 2}]) is False
