# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Command-line entry point: `plumbline score`, `gate`, `score-routing`.

All commands read CSV files; `score` and `gate` run against a SQLite database,
`score-routing` needs no database at all — so the whole tool works from a
fresh clone with no cloud. `score` grades predictions and prints execution +
judge-adjusted accuracy with confidence intervals; `gate` audits a gold set
and exits non-zero if any gold is flagged, so it can guard a CI step;
`score-routing` grades table-routing predictions (which table should answer
the question) with the same rate-plus-interval reporting. The LLM judge stays
opt-in (--judge-gemini) to keep the default path cloud-free.

CSV formats:
- predictions:         columns id, question, gold_sql, pred_sql
- golds:               columns id, gold_sql
- routing predictions: columns id, question, gold_table, pred_table
- catalog (routing):   plain text, one valid dataset.table per line,
                       blank lines and #-comments skipped
"""
from __future__ import annotations

import argparse
import csv

from .executor import SQLiteExecutor
from .gate import gate, summarize
from .harness import EvalReport, evaluate
from .routing import RoutingReport, evaluate_routing, normalize_table_name


def _load_predictions(path: str) -> list[tuple[str, str, str, str]]:
    """Read a predictions CSV into (id, question, gold_sql, pred_sql) tuples."""
    with open(path, newline="") as handle:
        return [
            (row["id"], row["question"], row["gold_sql"], row["pred_sql"])
            for row in csv.DictReader(handle)
        ]


def _load_golds(path: str) -> list[tuple[str, str]]:
    """Read a golds CSV into (id, gold_sql) tuples."""
    with open(path, newline="") as handle:
        return [(row["id"], row["gold_sql"]) for row in csv.DictReader(handle)]


def _load_routing_predictions(path: str) -> list[tuple[str, str, str, str]]:
    """Read a routing CSV into (id, question, gold_table, pred_table) tuples."""
    with open(path, newline="") as handle:
        return [
            (row["id"], row["question"], row["gold_table"], row["pred_table"])
            for row in csv.DictReader(handle)
        ]


def _load_catalog(path: str) -> set[str]:
    """Read a catalog file: one valid dataset.table per line.

    Blank lines and #-comments are skipped; each entry is normalized the same
    way as the predictions, so a 3-part or quoted catalog still matches.
    """
    with open(path) as handle:
        return {
            normalize_table_name(line)
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }


def _mark(result) -> str:
    """One-word correctness tag for a per-question line."""
    if result.execution_correct is None:
        return "excluded"
    if result.execution_correct:
        return "correct"
    if result.adjusted_correct:
        return "wrong -> upgraded"  # judge overrode the exact comparator
    return "wrong"


def format_score_report(report: EvalReport, confidence: float = 0.95) -> str:
    """Render an EvalReport as text: both rates with CIs, then a per-question list."""
    pct = round(confidence * 100)
    execution = report.execution
    adjusted = report.adjusted
    lines = [
        f"execution accuracy: {execution.point:.3f}  "
        f"{pct}% CI [{execution.low:.3f}, {execution.high:.3f}]  "
        f"({execution.k}/{execution.n})",
        f"judge-adjusted:     {adjusted.point:.3f}  "
        f"{pct}% CI [{adjusted.low:.3f}, {adjusted.high:.3f}]  "
        f"({adjusted.k}/{adjusted.n})",
        f"scored {report.n_scored} of {report.n_total}  "
        f"({report.n_excluded} excluded)",
        "",
        "per-question:",
    ]
    for result in report.results:
        lines.append(f"  {result.id}  {result.verdict:<10}  {_mark(result)}")
    return "\n".join(lines)


def format_gate_report(findings) -> str:
    """Render gate findings as text: a tally line, then the flagged golds."""
    clean = sum(1 for f in findings if f.ok)
    flagged = [f for f in findings if not f.ok]
    lines = [
        f"gate: {len(findings)} golds, {clean} clean, {len(flagged)} flagged"
    ]
    counts = summarize(findings)
    if counts:
        tally = "  ".join(f"{name}={n}" for name, n in sorted(counts.items()))
        lines.append(f"defects: {tally}")
    if flagged:
        # Pad the id to the widest flagged id so the defect column lines up.
        id_width = max(len(f.gold_id) for f in flagged)
        lines.append("")
        lines.append("flagged:")
        for finding in flagged:
            defects = ",".join(finding.defects)
            lines.append(
                f"  {finding.gold_id:<{id_width}}  {defects:<20}  {finding.detail}"
            )
    return "\n".join(lines)


def _routing_mark(result) -> str:
    """One-phrase correctness tag for a per-question routing line."""
    if result.table_correct:
        return "correct"
    if result.dataset_correct:
        return "dataset only"  # right dataset, wrong table
    return "wrong"


def format_routing_report(
    report: RoutingReport, confidence: float = 0.95, catalog_used: bool = False
) -> str:
    """Render a RoutingReport as text: both rates with CIs, then per-question.

    The snap line is printed whenever a catalog was in play — including
    "snapped 0" — and never without one, so its absence means snapping was
    off, not that nothing snapped.
    """
    pct = round(confidence * 100)
    dataset = report.dataset
    table = report.table
    lines = [
        f"dataset accuracy: {dataset.point:.3f}  "
        f"{pct}% CI [{dataset.low:.3f}, {dataset.high:.3f}]  "
        f"({dataset.k}/{dataset.n})",
        f"table accuracy:   {table.point:.3f}  "
        f"{pct}% CI [{table.low:.3f}, {table.high:.3f}]  "
        f"({table.k}/{table.n})",
    ]
    if catalog_used:
        lines.append(
            f"snapped {report.n_snapped} of {report.n_total} predictions "
            "to the catalog"
        )
    lines += ["", "per-question:"]
    for result in report.results:
        shown = result.pred_table or "(empty)"
        if result.snapped:
            shown = f"{normalize_table_name(result.raw_prediction)} -> {shown}"
        tag = _routing_mark(result)
        if result.snapped:
            tag = f"snapped, {tag}"
        lines.append(f"  {result.id}  {shown:<40}  {tag}")
    return "\n".join(lines)


def _cmd_score(args: argparse.Namespace) -> int:
    executor = SQLiteExecutor(args.db, row_cap=args.row_cap)
    predictions = _load_predictions(args.predictions)
    backend = None
    if args.judge_gemini:
        # Imported lazily so the default (no-judge) path never needs the extra.
        from .backends import gemini_backend

        backend = gemini_backend(model=args.judge_model, project=args.judge_project)
    report = evaluate(
        executor, predictions, judge_backend=backend, confidence=args.confidence
    )
    print(format_score_report(report, args.confidence))
    return 0


def _cmd_score_routing(args: argparse.Namespace) -> int:
    rows = _load_routing_predictions(args.predictions)
    catalog = _load_catalog(args.catalog) if args.catalog else None
    report = evaluate_routing(rows, catalog=catalog, confidence=args.confidence)
    print(
        format_routing_report(
            report, args.confidence, catalog_used=catalog is not None
        )
    )
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    executor = SQLiteExecutor(args.db, row_cap=args.row_cap)
    golds = _load_golds(args.golds)
    findings = gate(golds, executor, check_stability=args.check_stability)
    print(format_gate_report(findings))
    # Non-zero so a CI step fails when the answer key has defects.
    return 0 if all(f.ok for f in findings) else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plumbline",
        description="Execution-accuracy evaluation for text-to-SQL.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="grade predictions against gold SQL")
    score.add_argument("--db", required=True, help="path to the SQLite database")
    score.add_argument(
        "--predictions",
        required=True,
        help="CSV with columns id, question, gold_sql, pred_sql",
    )
    score.add_argument("--row-cap", type=int, default=None, help="max rows per query")
    score.add_argument(
        "--confidence", type=float, default=0.95, help="CI confidence level"
    )
    score.add_argument(
        "--judge-gemini",
        action="store_true",
        help="run the LLM judge on disagreements (needs the [gemini] extra)",
    )
    score.add_argument("--judge-model", default="gemini-2.5-pro")
    score.add_argument("--judge-project", default=None)
    score.set_defaults(func=_cmd_score)

    gate_p = sub.add_parser("gate", help="audit a gold set for defects")
    gate_p.add_argument("--db", required=True, help="path to the SQLite database")
    gate_p.add_argument(
        "--golds", required=True, help="CSV with columns id, gold_sql"
    )
    gate_p.add_argument("--row-cap", type=int, default=None, help="max rows per query")
    gate_p.add_argument(
        "--check-stability",
        action="store_true",
        help="run each gold twice to catch non-deterministic results (2x cost)",
    )
    gate_p.set_defaults(func=_cmd_gate)

    routing = sub.add_parser(
        "score-routing", help="grade table-routing predictions"
    )
    routing.add_argument(
        "--predictions",
        required=True,
        help="CSV with columns id, question, gold_table, pred_table",
    )
    routing.add_argument(
        "--catalog",
        default=None,
        help="file with one valid dataset.table per line; snaps near-miss "
        "predictions to the closest entry and reports how many",
    )
    routing.add_argument(
        "--confidence", type=float, default=0.95, help="CI confidence level"
    )
    routing.set_defaults(func=_cmd_score_routing)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args and dispatch; returns the process exit code."""
    args = _build_parser().parse_args(argv)
    return args.func(args)
