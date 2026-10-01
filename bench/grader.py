"""
The grader: open the case's database, run every check, report each one.

It never looks at the page, the screenshot, or what the agent *said*.
Only at what actually happened.
"""

from __future__ import annotations

from app.db import connect
from bench.model import Case, CheckResult, Grade


def grade(case: Case) -> Grade:
    conn = connect(case.db_path)
    results = []
    try:
        for check in case.task.checks:
            try:
                passed, detail = check.fn(conn, case.baseline)
            except Exception as e:  # a crashing check is a failed check, never a pass
                passed, detail = False, f"check crashed: {type(e).__name__}: {e}"
            results.append(CheckResult(check.name, bool(passed), detail))
    finally:
        conn.close()
    return Grade(task_id=case.task.id, results=tuple(results))
