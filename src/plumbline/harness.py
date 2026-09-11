# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""End-to-end scoring: tie executor + comparator + judge + metrics together.

`evaluate` is the one call a user makes. For each prediction it runs the gold
and predicted SQL through the injected Executor, compares the result sets, and
(optionally) sends the leftover disagreements to the LLM judge. It returns an
EvalReport carrying every per-question outcome plus two headline rates, each
with a Wilson confidence interval:

- execution accuracy   — the exact comparator alone
- judge-adjusted       — after the one-sided judge upgrades EQUIVALENT cases;
                         always >= execution accuracy

Two outcomes are excluded from the denominator because they carry no signal
about the prediction:
- EMPTY_BOTH  — gold and prediction both empty. The gold should have been
                removed by the gate first; counting it either way is
                misleading (a model that returns nothing would "match").
- GOLD_ERROR  — the gold query itself failed to run; it can't grade anything.
A failed PREDICTION, by contrast, is a real wrong answer and counts as
incorrect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .compare import compare
from .executor import Executor
from .judge import judge as judge_call
from .metrics import Proportion, proportion

# Verdicts beyond the comparator's own (MATCH/MISMATCH/EMPTY_ONE/EMPTY_BOTH).
GOLD_ERROR = "GOLD_ERROR"
PRED_ERROR = "PRED_ERROR"

# A prediction tuple: (id, question, gold_sql, predicted_sql).
Prediction = tuple[str, str, str, str]


@dataclass(frozen=True)
class QuestionResult:
    """The outcome of scoring one prediction.

    `execution_correct` / `adjusted_correct` are None when the question is
    excluded from the rates (EMPTY_BOTH or GOLD_ERROR), True/False otherwise.
    `adjusted_correct` differs from `execution_correct` only when the judge
    upgraded a disagreement. `judge_verdict` and `detail` are for an audit log.
    """

    id: str
    question: str
    verdict: str
    execution_correct: bool | None
    adjusted_correct: bool | None
    judged: bool
    judge_verdict: str
    detail: str


@dataclass(frozen=True)
class EvalReport:
    """All per-question results plus the two headline rates with CIs.

    `n_scored` is the denominator (excluded questions removed). `execution` and
    `adjusted` are Proportions, so a caller always has the interval next to the
    rate. Use `correctness_vector()` to feed metrics.mcnemar_from_vectors when
    comparing two runs.
    """

    results: list[QuestionResult]
    n_total: int
    n_scored: int
    n_excluded: int
    execution: Proportion
    adjusted: Proportion

    def correctness_vector(self, adjusted: bool = False) -> list[bool]:
        """Per-question correctness over the SCORED questions, in input order.

        Excluded questions (None) are dropped. For McNemar, run the same
        predictions in the same order through both systems and the two vectors
        line up by position — the alignment mcnemar_from_vectors requires. Pass
        adjusted=True to compare judge-adjusted correctness instead of raw
        execution.
        """
        field = "adjusted_correct" if adjusted else "execution_correct"
        return [
            getattr(r, field) for r in self.results if getattr(r, field) is not None
        ]


def _evaluate_one(
    executor: Executor,
    prediction: Prediction,
    judge_backend: Callable[[str], str] | None,
) -> QuestionResult:
    """Score a single prediction; never raises (errors become verdicts)."""
    id_, question, gold_sql, pred_sql = prediction

    gold = executor.execute(gold_sql)
    if not gold.ok:
        # The answer key is broken — can't grade anything against it. Excluded.
        return QuestionResult(
            id_, question, GOLD_ERROR, None, None, False, "",
            f"gold failed: {gold.error}",
        )

    pred = executor.execute(pred_sql)
    if not pred.ok:
        # A prediction that won't run is a real wrong answer, not an exclusion.
        return QuestionResult(
            id_, question, PRED_ERROR, False, False, False, "",
            f"prediction failed: {pred.error}",
        )

    verdict = compare(gold.rows, pred.rows, gold_sql)

    if verdict == "EMPTY_BOTH":
        return QuestionResult(
            id_, question, verdict, None, None, False, "", "both empty — excluded"
        )
    if verdict == "MATCH":
        return QuestionResult(id_, question, verdict, True, True, False, "", "")

    # MISMATCH or EMPTY_ONE: wrong by the exact comparator. The one-sided judge
    # may still upgrade it, but only to MATCH, and only if a backend was given.
    if judge_backend is None:
        return QuestionResult(id_, question, verdict, False, False, False, "", "")

    jv = judge_call(question, gold_sql, pred_sql, gold.rows, pred.rows, judge_backend)
    return QuestionResult(
        id_, question, verdict,
        execution_correct=False,
        adjusted_correct=jv.equivalent,
        judged=True,
        judge_verdict=jv.verdict,
        detail=jv.reason,
    )


def evaluate(
    executor: Executor,
    predictions: list[Prediction],
    judge_backend: Callable[[str], str] | None = None,
    confidence: float = 0.95,
) -> EvalReport:
    """Score every prediction and return an EvalReport.

    Args:
        executor: runs both gold and predicted SQL (SQLite, BigQuery, or any
            object with .execute).
        predictions: list of (id, question, gold_sql, predicted_sql).
        judge_backend: optional Callable[[str], str]; when given, the judge runs
            on each MISMATCH/EMPTY_ONE and can upgrade it to correct. When None,
            adjusted accuracy equals execution accuracy.
        confidence: confidence level for both Wilson intervals (default 0.95).
    """
    results = [_evaluate_one(executor, p, judge_backend) for p in predictions]

    scored = [r for r in results if r.execution_correct is not None]
    n_scored = len(scored)
    k_exec = sum(1 for r in scored if r.execution_correct)
    k_adjusted = sum(1 for r in scored if r.adjusted_correct)

    return EvalReport(
        results=results,
        n_total=len(results),
        n_scored=n_scored,
        n_excluded=len(results) - n_scored,
        execution=proportion(k_exec, n_scored, confidence),
        adjusted=proportion(k_adjusted, n_scored, confidence),
    )
