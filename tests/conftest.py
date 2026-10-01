from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from bench.env import build_seed, prepare_case
from bench.grader import grade


@pytest.fixture(scope="session")
def seed(tmp_path_factory) -> Path:
    # Built once per test session; every test gets its own copy.
    return build_seed(tmp_path_factory.mktemp("seed") / "seed.db")


@pytest.fixture
def run_case(seed, tmp_path):
    """run_case(task, agent) -> Grade, on a fresh isolated database."""
    def _run(task, agent):
        case = prepare_case(task, tmp_path / task.id, seed)
        with TestClient(create_app(case.db_path)) as client:
            agent(client)
        return grade(case)
    return _run


@pytest.fixture
def shop(seed, tmp_path):
    """A logged-out client on a fresh copy of the seed, plus its db path."""
    import shutil
    db = tmp_path / "shop.db"
    shutil.copyfile(seed, db)
    with TestClient(create_app(db)) as client:
        yield client, db
