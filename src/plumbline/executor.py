# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Database backends behind one interface.

The harness, gate, and judge only ever need "run this SQL, give me rows".
Where the rows come from — SQLite for examples/tests, BigQuery in production —
is hidden behind the Executor protocol, so the core stays pure stdlib and the
cloud backend lives behind an optional extra.

Safety rail shared by every backend: predicted SQL is untrusted model output,
so a statement that isn't read-only is rejected before it ever reaches the
database. For BigQuery the same guard rides along with a dry-run that enforces
maximum_bytes_billed and disables the query cache.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ExecResult:
    """Outcome of running one SQL statement.

    `error is None` means the statement ran. An empty `rows` with no error is
    a real zero-row result (a signal the gate cares about), which is why
    "ran but empty" and "failed to run" are kept distinct rather than both
    collapsing to []. `truncated` is True when a row cap cut the result off.
    """
    rows: list[dict] = field(default_factory=list)  # comparator Row shape
    error: str | None = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        """True if the statement executed (even if it returned zero rows)."""
        return self.error is None


@runtime_checkable
class Executor(Protocol):
    """Anything that can run a SQL string and return an ExecResult.

    Structural: a class satisfies this by having a matching execute method,
    with no import of or inheritance from this module — so the BigQuery
    backend in the extra and any user-supplied backend both qualify.
    """

    def execute(self, sql: str) -> ExecResult:
        ...


# Statements that modify data or schema. A read-only query must not begin with
# these, and (to block data-modifying CTEs like WITH t AS (DELETE ...)) must
# not contain them anywhere. The trade-off: a SELECT with the literal word
# DROP inside a string is conservatively rejected. For untrusted model output
# that's the safe direction.
_WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|REPLACE|"
    r"GRANT|REVOKE|ATTACH|DETACH|VACUUM|PRAGMA)\b",
    re.IGNORECASE,
)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_comments(sql: str) -> str:
    """Remove -- line and /* */ block comments, returning the bare SQL text.

    Shared by the read-only guard and the gold gate: both need to reason about
    the actual statement, not commented-out text (a DROP in a comment is inert,
    a gold that is *only* a comment is an empty stub).
    """
    return _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql or ""))


def is_read_only(sql: str) -> bool:
    """True if `sql` is a single read-only statement (starts with SELECT/WITH).

    Comments are stripped first; a trailing semicolon is allowed but an
    internal one (multiple statements) is rejected, so "SELECT 1; DROP TABLE t"
    can't slip a write past the first-keyword check.
    """
    s = strip_comments(sql).strip().rstrip(";").strip()
    if not s or ";" in s:
        return False
    first = s.split(None, 1)[0].upper()
    if first not in ("SELECT", "WITH"):
        return False
    return not _WRITE_KEYWORDS.search(s)


class SQLiteExecutor:
    """Executor over a stdlib sqlite3 database — the core/example/test backend.

    Connects per call (stateless, idempotent) and returns rows as dicts. Set
    row_cap to bound a result set; the executor fetches one extra row to tell
    whether the cap actually truncated. enforce_read_only rejects non-SELECT
    statements before touching the DB.
    """

    def __init__(
        self,
        database: str,
        row_cap: int | None = None,
        enforce_read_only: bool = True,
    ):
        self.database = database
        self.row_cap = row_cap
        self.enforce_read_only = enforce_read_only

    def execute(self, sql: str) -> ExecResult:
        if self.enforce_read_only and not is_read_only(sql):
            return ExecResult(error="rejected: statement is not read-only")
        con = sqlite3.connect(self.database)
        con.row_factory = sqlite3.Row
        try:
            cur = con.execute(sql)
            if self.row_cap is None:
                fetched = cur.fetchall()
                truncated = False
            else:
                fetched = cur.fetchmany(self.row_cap + 1)
                truncated = len(fetched) > self.row_cap
                fetched = fetched[: self.row_cap]
            return ExecResult(rows=[dict(r) for r in fetched], truncated=truncated)
        except sqlite3.Error as e:
            # Wrap the driver error as a string so callers stay backend-agnostic
            # and a bad gold/pred query becomes EXEC_FAIL, never a crash.
            return ExecResult(error=f"{type(e).__name__}: {e}")
        finally:
            con.close()


class BigQueryExecutor:
    """Executor over BigQuery — optional, behind the [bigquery] extra.

    Not exercised by the fast test suite (needs credentials). Every query is
    dry-run first to enforce a byte ceiling and confirm it is read-only, then
    run with the cache off so repeated eval runs measure the engine, not a
    cached result. google-cloud-bigquery is imported lazily so importing this
    module never requires the extra.
    """

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        maximum_bytes_billed: int = 1_000_000_000,  # 1 GB ceiling by default
        row_cap: int | None = None,
    ):
        from google.cloud import bigquery  # lazy: keeps core install stdlib-only

        self._bq = bigquery
        self.client = bigquery.Client(project=project, location=location)
        self.location = location
        self.maximum_bytes_billed = maximum_bytes_billed
        self.row_cap = row_cap

    def execute(self, sql: str) -> ExecResult:
        if not is_read_only(sql):
            return ExecResult(error="rejected: statement is not read-only")
        bq = self._bq
        try:
            dry = self.client.query(
                sql,
                job_config=bq.QueryJobConfig(dry_run=True, use_query_cache=False),
                location=self.location,
            )
            scanned = dry.total_bytes_processed or 0
            if scanned > self.maximum_bytes_billed:
                return ExecResult(
                    error=f"rejected: dry-run scans {scanned} bytes > "
                    f"ceiling {self.maximum_bytes_billed}"
                )
            job = self.client.query(
                sql,
                job_config=bq.QueryJobConfig(
                    use_query_cache=False,
                    maximum_bytes_billed=self.maximum_bytes_billed,
                ),
                location=self.location,
            )
            rows = [dict(r) for r in job.result()]
        except Exception as e:  # google client raises many exc types; flatten
            return ExecResult(error=f"{type(e).__name__}: {e}")
        truncated = self.row_cap is not None and len(rows) > self.row_cap
        if truncated:
            rows = rows[: self.row_cap]
        return ExecResult(rows=rows, truncated=truncated)
