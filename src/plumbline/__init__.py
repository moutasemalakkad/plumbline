# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""plumbline — executor-agnostic text-to-SQL evaluation.

Public surface (built incrementally):
- compare.compare      execution / denotation match between two result sets
- metrics.*            proportion with Wilson interval, McNemar paired test
- executor.Executor    protocol; SQLiteExecutor (core), BigQueryExecutor (extra)
- backends.*           echo_backend (tests), gemini_backend (extra) for the judge
- judge.judge          FLEX-style LLM equivalence tiebreaker (backend injected)
- gate.*               gold-set validation gate
- harness.evaluate     end-to-end scoring over (id, question, gold_sql, pred_sql)
- routing.evaluate_routing  table-routing accuracy over (id, question,
                       gold_table, pred_table), optional catalog snapping
"""

__version__ = "0.1.0"
