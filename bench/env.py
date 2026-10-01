"""
Environment management: build the clean world once, copy it per case.

This is the isolation guarantee. Every case gets its own database file,
so no case can see another's cart, orders, or stock — even when many run
at the same time in v1.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.db import connect, init_db
from bench.model import Baseline, Case, Task


def build_seed(path: str | Path) -> Path:
    """Build the pristine seed database. Slow-ish (password hashing), so do it once."""
    return init_db(path)


def capture_baseline(conn) -> Baseline:
    max_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM orders").fetchone()[0]
    return Baseline(max_order_id=max_id)


def prepare_case(task: Task, workdir: str | Path, seed: str | Path) -> Case:
    """
    Fresh copy of the seed -> task setup -> baseline snapshot.

    After this returns, the database is exactly the world the agent should
    start in, and the baseline records what existed before it touched anything.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / f"{task.id}.db"
    shutil.copyfile(seed, db_path)

    conn = connect(db_path)
    try:
        with conn:
            task.setup(conn)
        baseline = capture_baseline(conn)
    finally:
        conn.close()

    return Case(task=task, db_path=db_path, baseline=baseline)
