# Methodology

Why this library measures what it measures, the alternatives rejected, and the
known limitations. The short version lives in the [README](../README.md); this is
the reference.

## 1. Execution accuracy, not string match

A text-to-SQL system is graded by *running* its SQL and comparing the result rows
to the gold query's rows — not by comparing SQL text. Two queries written
differently (a `JOIN` vs a subquery, a different column order, an extra harmless
`ORDER BY`) can return the same answer, and string match would wrongly call them
different. This is the metric used by Spider (Yu et al., EMNLP 2018) and BIRD (Li
et al., NeurIPS 2023).

The comparator (`compare.py`) is adapted from the reference implementation in
`taoyds/test-suite-sql-eval` (Zhong, Yu, Klein, EMNLP 2020,
https://aclanthology.org/2020.emnlp-main.29/).

## 2. Comparator semantics

Four rules, each tracking a fact about SQL:

| rule | why | rejected alternative |
|---|---|---|
| compare the **full** result set | a first-N sample produces false MISMATCH when both queries return many rows in different orders | sampling rows |
| **column order ignored** (sort each row's values before tupling) | `SELECT a, b` and `SELECT b, a` denote the same answer; O(c log c) per row | enumerating O(c!) column permutations |
| **multiset** semantics, switching to **positional** when the gold has `ORDER BY` | an unordered query is a set; an ordered query is a sequence | always-set (misses a broken `ORDER BY`/`LIMIT`); always-positional (false MISMATCH on unordered queries) |
| **floats rounded to 4 decimals** | `AVG`/`STDDEV` jitter across engines and across runs | exact float equality |

NULL is carried as a sentinel string so rows sort and hash cleanly; `NULL` is not
equal to `0` or to empty string.

What the comparator deliberately does **not** do:
- **Type coercion** — `1` and `"1"` compare as different (Spider matches this).
- **Column-name matching** — names are ignored; only values are compared.

## 3. Exclusion rules

Two outcomes are dropped from the denominator because they carry no signal about a
prediction:
- **EMPTY_BOTH** — gold and prediction both return nothing. Counting it as correct
  would reward a model that returns nothing; counting it wrong punishes a correct
  empty answer. The gate (§5) should remove empty golds first, so this is rare.
- **GOLD_ERROR** — the gold query itself failed to run; it can't grade anything.

A failed *prediction* is different: it is a real wrong answer and counts as
incorrect.

## 4. The LLM judge is one-sided — read this before trusting a number

Single-database execution match has a false-negative rate: two genuinely
equivalent result sets can differ in a way the exact comparator can't reconcile
(a full name vs a first name, a representation difference). The test-suite paper
(EMNLP 2020) reports this is roughly 2.5% on average and up to ~8.1% in the worst
case for single-DB denotation match.

The optional judge is the cheap partial fix from FLEX (Kim et al., NAACL 2025,
https://arxiv.org/abs/2409.19014): an LLM looks at the cases the comparator
already flagged wrong and decides whether they are actually equivalent.

**The limitation, stated plainly:** the judge is one-sided. It runs only on the
MISMATCH/EMPTY_ONE residual, and only an `EQUIVALENT` verdict upgrades a row to
correct. It never re-checks a MATCH. So it can only **raise** the reported score,
never lower it. A judge-adjusted number is therefore an *upper-leaning* estimate.

For a headline number you defend, do a **two-sided human audit**: sample MATCHes
too (the judge never sees those), and report inter-rater agreement (Cohen's kappa)
between the human and the automated verdict. The judge reduces false negatives; it
does not certify the score.

The judge is also **fail-safe**: any backend error (timeout, bad key, quota)
returns an empty verdict and leaves the row wrong, never raises. One dead call must
not take down a long run.

## 5. The gold-set gate audits the oracle

Every accuracy number is computed against the gold queries, so a broken gold
silently corrupts the metric. The gate (`gate.py`) runs each gold through an
executor and flags:

| defect | meaning |
|---|---|
| `NON_SELECT` | not a read-only SELECT/WITH |
| `PLACEHOLDER_COMMENT` | empty after stripping comments, or carries a TODO/FIXME marker |
| `EXEC_FAIL` | raises when executed |
| `ZERO_ROWS` | returns no rows (often a wrong filter or stale data) |
| `DEGENERATE` | a single trivial cell (NULL or 0) |
| `TRUNCATED` | hit the executor row cap |
| `UNSTABLE` | two runs returned different rows (opt-in; doubles cost) |

`DEGENERATE` is distinct from `ZERO_ROWS` for a reason worth stating: an aggregate
over an empty set returns *one* row, not zero. `SELECT SUM(x) FROM t WHERE <never>`
yields `[{None}]`, which `ZERO_ROWS` would miss — `DEGENERATE` catches the
non-answer.

## 6. Routing accuracy scores the table-picking stage on its own

In a warehouse with many tables, a text-to-SQL pipeline usually has a routing
stage first: pick the one `dataset.table` that should answer the question, then
write SQL against it. A routing miss guarantees a generation miss, so folding
both stages into one end-to-end number hides *which* stage failed. `score-routing`
grades routing separately, as single-label top-1 classification.

**Two tiers, not one.** Most routing misses land in the right dataset but the
wrong table; a single exact-match rate hides that, and the fix differs (better
table descriptions vs. better dataset scoping). So the report carries both
**dataset accuracy** (the part before the first dot matches) and **table
accuracy** (exact `dataset.table` match).

**Normalization before comparison.** Model output arrives with decoration that
should not count as a miss: surrounding quotes or backticks, a `bigquery:`
prefix, a 3-part `project.dataset.table`. Both sides are reduced to a canonical
2-part name; comparison stays case-sensitive because warehouse table names are.

**Empty predictions are misses, never exclusions.** The SQL harness excludes
GOLD_ERROR and EMPTY_BOTH because they carry no signal about the prediction
(§3). Routing has no analogue: a router that returns nothing gave a real wrong
answer, so the denominator is always the full question count. A broken routing
*gold* raises instead of being excluded — the gold is a single table name and
cheap to fix.

**Catalog snapping is opt-in and always reported.** A model that generates the
table name freely can blend two real names into one that does not exist. With
`--catalog` (the set of valid names), a near-miss prediction snaps to the
closest valid name (difflib similarity, cutoff 0.8) before scoring. This
recovers the model's evident intent but raises the number, so the report always
prints the snapped count, and the flag defaults to off. Two runs are only
comparable when they agree on whether a catalog was used — the rejected
alternative, always-on snapping, produces numbers that silently disagree with
un-snapped ones.

## 7. Statistics: Wilson interval and McNemar's test

An accuracy like 19/35 is a point estimate from a noisy process; a 35-question
sample can barely pin it down. So every rate ships with an interval, and two
systems are compared with a paired test rather than by eyeballing which number is
bigger.

- **Wilson score interval** (Wilson, 1927) instead of the normal approximation
  `p ± z·√(p(1−p)/n)`. The normal approximation produces impossible bounds (<0 or
  >1) near the extremes and is too narrow at small n; Wilson inverts the score test
  and stays inside [0, 1].
- **McNemar's exact test** (McNemar, 1947) for comparing two systems on the same
  questions. Only the discordant pairs (one system right, the other wrong) carry
  signal; the exact binomial is used (not the chi-square approximation) because
  eval disagreement counts are usually small.

**Why hand-rolled, not statsmodels/scipy:** `statsmodels.proportion_confint(method=
"wilson")` and `contingency_tables.mcnemar(exact=True)` give identical numbers, and
for a one-off script they'd be the right call. But this is a library others install,
and statsmodels pulls scipy + numpy + pandas to provide ~40 lines of math — which
would make `make test` and `make example` require the whole scientific stack from a
fresh clone (and in some locked-down environments that stack isn't reachable). The
pure-stdlib core keeps the install story to "Python 3.10+". The risk of hand-rolled
stats is subtly-wrong math, so the numbers are pinned to a published reference
(Wilson 19/35 → [0.382, 0.695]) and an optional parity test asserts equality with
statsmodels when it happens to be installed. The inverse-normal CDF is Acklam's
rational approximation (accurate to ~1.15e-9), so no SciPy is needed for the z
multiplier.

## 8. Deliberately skipped

- **Test-suite accuracy** (perturbed database copies, EMNLP 2020) — the rigorous
  fix for the false-negative rate in §4. It needs many DB copies per question and
  is out of scope; the one-sided judge is the cheap partial mitigation.
- **Type coercion and column-name matching** in the comparator (§2).
- **Recall@k and MRR for routing** (§6) — those grade a ranked candidate list;
  this scorer grades the single table the router committed to. If the pipeline
  retrieves candidates before picking, retrieval recall is a separate
  measurement.
- **Multi-label routing golds** — a question answerable from two tables scores
  as a miss unless the gold names the picked one. One gold table per question
  keeps the answer key auditable; questions that genuinely have two right
  answers belong in the gold set twice or not at all.

## References

- Yu et al. Spider: A Large-Scale Human-Labeled Dataset for Complex and
  Cross-Domain Semantic Parsing and Text-to-SQL Task. EMNLP 2018.
- Li et al. Can LLM Already Serve as a Database Interface? A Big Bench for
  Large-Scale Database Grounded Text-to-SQLs (BIRD). NeurIPS 2023.
- Zhong, Yu, Klein. Semantic Evaluation for Text-to-SQL with Distilled Test Suites.
  EMNLP 2020. https://aclanthology.org/2020.emnlp-main.29/
- Kim et al. FLEX: Expert-level False-Less EXecution Metric for Reliable
  Text-to-SQL Benchmark. NAACL 2025. https://arxiv.org/abs/2409.19014
- Wilson, E. B. Probable Inference, the Law of Succession, and Statistical
  Inference. Journal of the American Statistical Association, 1927.
- McNemar, Q. Note on the Sampling Error of the Difference Between Correlated
  Proportions or Percentages. Psychometrika, 1947.
