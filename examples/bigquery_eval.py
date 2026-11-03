# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Score a text-to-SQL system against BigQuery instead of the SQLite example.

The CLI takes `--db`, a SQLite file, so a warehouse run goes through the Python
API: swap SQLiteExecutor for BigQueryExecutor and everything downstream -- the
comparator, the gate, the judge, the Wilson intervals -- is unchanged. That is
the point of the Executor protocol: the executor is the only cloud-aware part.

Needs the extra and credentials:

    uv pip install -e '.[bigquery]'        # add ,gemini for the judge
    gcloud auth application-default login
    python examples/bigquery_eval.py --project my-project --predictions preds.csv

The predictions CSV is the same four columns the SQLite example uses:
`id,question,gold_sql,predicted_sql`. Point the SQL at real tables.

Two BigQuery-specific things worth knowing before the first run:

- **Every query is dry-run first against a byte ceiling.** The default here is
  10 GB, higher than the library's 1 GB, because a production fact table scans
  more than a toy schema: an unfiltered `SELECT MAX(snapshot_date)` on a daily
  snapshot table can read a terabyte. A rejection prints the real figure, so
  raise the ceiling deliberately rather than removing it.
- **Scoring runs each query twice**, once for the gold and once for the
  prediction, and the cache is off on purpose so a rerun measures the engine
  rather than a cached answer. Budget for that: cost is roughly
  2 x questions x bytes-per-query.
"""
from __future__ import annotations

import argparse
import csv
import sys

from plumbline.executor import BigQueryExecutor
from plumbline.gate import gate, summarize
from plumbline.harness import evaluate


def read_predictions(path: str) -> list[tuple[str, str, str, str]]:
    """Read id,question,gold_sql,predicted_sql rows into the harness's tuple."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    missing = {"id", "question", "gold_sql", "predicted_sql"} - set(rows[0] or {})
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    return [(r["id"], r["question"], r["gold_sql"], r["predicted_sql"]) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True, help="billing project for the queries")
    ap.add_argument("--location", default="US", help="dataset location (default US)")
    ap.add_argument("--predictions", required=True,
                    help="CSV: id,question,gold_sql,predicted_sql")
    ap.add_argument("--max-bytes", type=int, default=10_000_000_000,
                    help="per-query dry-run ceiling in bytes (default 10 GB)")
    ap.add_argument("--row-cap", type=int, default=None,
                    help="truncate result sets beyond this many rows")
    ap.add_argument("--judge", action="store_true",
                    help="run the LLM equivalence judge on mismatches "
                         "(needs the [gemini] extra)")
    ap.add_argument("--gate-only", action="store_true",
                    help="audit the gold queries and stop, before spending on scoring")
    args = ap.parse_args()

    executor = BigQueryExecutor(
        project=args.project,
        location=args.location,
        maximum_bytes_billed=args.max_bytes,
        row_cap=args.row_cap,
    )
    predictions = read_predictions(args.predictions)

    # Audit the answer key first. A gold that errors or returns nothing makes
    # every score computed against it meaningless, and finding that out after
    # paying for a full run is the expensive way to learn it.
    findings = gate([(pid, gold) for pid, _, gold, _ in predictions], executor)
    defects = summarize(findings)
    print(f"gate: {len(predictions)} golds checked, "
          f"{sum(defects.values())} defects {defects or '{}'}")
    for f in findings:
        if f.defects:
            # print detail, not just the class: the commonest BigQuery defect is
            # the byte ceiling, and "EXEC_FAIL" alone sends a reader hunting for
            # a SQL bug when the fix is --max-bytes
            print(f"  {f.gold_id}: {', '.join(f.defects)}"
                  + (f" -- {f.detail}" if f.detail else ""))
    if args.gate_only:
        return 0

    judge_backend = None
    if args.judge:
        from plumbline.backends import gemini_backend
        judge_backend = gemini_backend(project=args.project)

    report = evaluate(executor, predictions, judge_backend=judge_backend)

    print(f"\nscored {report.n_scored} of {report.n_total} "
          f"({report.n_excluded} excluded)")
    print(f"  execution accuracy {report.execution.point:.1%}  "
          f"[{report.execution.low:.1%}, {report.execution.high:.1%}]")
    if judge_backend:
        print(f"  judge-adjusted     {report.adjusted.point:.1%}  "
              f"[{report.adjusted.low:.1%}, {report.adjusted.high:.1%}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
