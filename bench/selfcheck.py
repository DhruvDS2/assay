"""
Calibrate the instrument before trusting it.

For every task, run two fake agents on fresh databases:
    oracle  does the task perfectly      -> every check must PASS
    noop    does nothing at all          -> at least one check must FAIL

If either expectation breaks, the bench is lying and no agent result
means anything yet. Run this after touching any task or the app.

    python -m bench.selfcheck
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from bench.env import build_seed, prepare_case
from bench.grader import grade
from bench.model import Grade, Task
from bench.tasks import TASKS


def noop(client) -> None:
    """The laziest possible agent. It should fail every task."""


def run_agent(task: Task, agent, workdir: Path, seed: Path) -> Grade:
    case = prepare_case(task, workdir, seed)
    with TestClient(create_app(case.db_path)) as client:
        agent(client)
    return grade(case)


def main() -> int:
    healthy = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        seed = build_seed(tmp / "seed.db")

        print(f"{'task':<22} {'oracle':<8} noop")
        print("-" * 60)
        for task in TASKS:
            o = run_agent(task, task.oracle, tmp / "oracle", seed)
            n = run_agent(task, noop, tmp / "noop", seed)

            o_txt = "PASS" if o.passed else "BROKEN"
            n_txt = f"fail ({', '.join(n.failed_checks)})" if not n.passed else "BROKEN (passed!)"
            print(f"{task.id:<22} {o_txt:<8} {n_txt}")

            if not o.passed:
                healthy = False
                for r in o.results:
                    if not r.passed:
                        print(f"    oracle failed {r.name}: {r.detail}")
            if n.passed:
                healthy = False
                print("    doing nothing passed this task — its checks are too weak")

    print("-" * 60)
    print("calibrated: grader separates success from failure" if healthy
          else "NOT calibrated — fix the tasks above before running any agent")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
