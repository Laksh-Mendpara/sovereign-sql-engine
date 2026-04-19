"""Helpers for validating and capping generated SQL."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SQL_FORBIDDEN_PATTERNS = (
    r"\binsert\b",
    r"\bupdate\b",
    r"\bdelete\b",
    r"\bdrop\b",
    r"\balter\b",
    r"\bcreate\b",
    r"\btruncate\b",
    r"\breplace\b",
    r"\bmerge\b",
    r"\bgrant\b",
    r"\brevoke\b",
    r"\battach\b",
    r"\bdetach\b",
    r"\bpragma\b",
    r"\bvacuum\b",
    r"\bcommit\b",
    r"\brollback\b",
)

_SQL_LIMIT = 30


@dataclass(frozen=True)
class SqlExecutionPlan:
    execution_sql: str | None
    remark: str
    blocked_by_firewall: bool


def is_read_only_sql(sql: str | None) -> bool:
    if not sql:
        return False

    normalized = sql.strip().rstrip(";").strip()
    lowered = normalized.lower()

    if not lowered.startswith(("select", "with", "explain")):
        return False

    return not any(re.search(pattern, lowered) for pattern in _SQL_FORBIDDEN_PATTERNS)


def apply_limit_30(sql: str | None) -> str | None:
    if not is_read_only_sql(sql):
        return None

    normalized = sql.strip().rstrip(";").strip()
    limit_match = re.search(r"\blimit\s+\d+\s*$", normalized, re.IGNORECASE)
    if limit_match:
        return re.sub(r"\blimit\s+\d+\s*$", "LIMIT 30", normalized, flags=re.IGNORECASE)

    return f"{normalized} LIMIT {_SQL_LIMIT}"


def plan_sql_execution(sql: str | None) -> SqlExecutionPlan:
    """Create the execution SQL and the user-facing remark for SSE."""
    if not sql:
        return SqlExecutionPlan(
            execution_sql=None,
            remark="No executable SQL was generated.",
            blocked_by_firewall=False,
        )

    normalized = sql.strip().rstrip(";").strip()

    if not is_read_only_sql(normalized):
        return SqlExecutionPlan(
            execution_sql=None,
            remark="Query blocked by firewall: write-related SQL.",
            blocked_by_firewall=True,
        )

    limit_match = re.search(r"\blimit\s+(\d+)\s*$", normalized, re.IGNORECASE)
    if not limit_match:
        return SqlExecutionPlan(
            execution_sql=f"{normalized} LIMIT 30",
            remark="Query limited to LIMIT 30 for execution.",
            blocked_by_firewall=False,
        )

    limit_value = int(limit_match.group(1))
    if limit_value > 60:
        return SqlExecutionPlan(
            execution_sql=None,
            remark=f"Query blocked by firewall: LIMIT {limit_value} exceeds 60.",
            blocked_by_firewall=True,
        )

    execution_sql = re.sub(r"\blimit\s+\d+\s*$", "LIMIT 39", normalized, flags=re.IGNORECASE)
    return SqlExecutionPlan(
        execution_sql=execution_sql,
        remark="Query limited to LIMIT 39 for execution.",
        blocked_by_firewall=False,
    )
