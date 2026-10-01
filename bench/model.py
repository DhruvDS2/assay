"""
The vocabulary of the bench. Everything else is built from these five types.

    Task      what the agent is asked to do, and how we know it did it
    Check     one named yes/no question about the database afterward
    Baseline  a snapshot taken *before* the agent runs
    Case      one task, on one fresh database, ready to run
    Grade     the answers to every check for one case
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Baseline:
    # Anything with an id above this was created by the agent.
    # This is how side-effect checks tell "new" from "already there".
    max_order_id: int


CheckFn = Callable[[sqlite3.Connection, Baseline], "tuple[bool, str]"]


@dataclass(frozen=True)
class Check:
    name: str      # short, stable, shows up in reports: "exactly_one_new_order"
    fn: CheckFn    # returns (passed, human-readable detail)


@dataclass(frozen=True)
class Task:
    id: str
    goal: str                                        # what the agent is told to do
    email: str                                       # account the agent logs in as
    password: str
    setup: Callable[[sqlite3.Connection], None]      # puts the world in its start state
    checks: tuple[Check, ...]                        # ALL must pass for the task to pass
    oracle: Callable[[Any], None]                    # scripted perfect run; proves the task is doable

    def instruction(self, base_url: str) -> str:
        """The exact text an agent receives."""
        return (
            f"You are using an online shop at {base_url}. "
            f"Log in with email {self.email} and password {self.password}. "
            f"Then: {self.goal}"
        )


@dataclass(frozen=True)
class Case:
    task: Task
    db_path: Path
    baseline: Baseline


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Grade:
    task_id: str
    results: tuple[CheckResult, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def failed_checks(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]
