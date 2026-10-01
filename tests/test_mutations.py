"""
The mutations perturb the *surface* and nothing else.

A mutation that silently changed a form action, an input name, or a SKU would
quietly make the grader lie — a task would fail not because the agent erred but
because the mutated page posts to the wrong place. These tests pin the contract:

- every task's oracle still passes under every mutation (achievable + contract),
- the server-read bits (actions, names, product/order hrefs) are byte-identical,
- and each mutation actually changes *something* an agent sees (no no-ops).
"""

from __future__ import annotations

import re
import shutil

import pytest
from fastapi.testclient import TestClient

from app.mutations import MUTATION_NAMES, MUTATIONS, create_mutated_app, mutate_app
from app.main import create_app
from bench.env import prepare_case
from bench.grader import grade
from bench.tasks import ALICE, TASKS, add, login

# What the server actually reads off a page — if any of this drifts, grading lies.
_ACTIONS = re.compile(r'action="([^"]+)"')
_NAMES = re.compile(r'name="([^"]+)"')
_REFS = re.compile(r'href="(/product/[^"]+|/orders/\d+)"')


def _contract(html: str) -> tuple[set, set, set]:
    return (set(_ACTIONS.findall(html)),
            set(_NAMES.findall(html)),
            set(_REFS.findall(html)))


@pytest.fixture
def fresh_db(seed, tmp_path):
    db = tmp_path / "shop.db"
    shutil.copyfile(seed, db)
    return db


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
@pytest.mark.parametrize("mutation", MUTATION_NAMES)
def test_oracle_passes_under_every_mutation(task, mutation, seed, tmp_path):
    """The oracle drives real endpoints; if a mutation broke one, a check fails."""
    case = prepare_case(task, tmp_path / f"{task.id}-{mutation}", seed)
    app = create_mutated_app(case.db_path, mutations=(mutation,))
    with TestClient(app) as client:
        task.oracle(client)
    g = grade(case)
    assert all(r.passed for r in g.results), \
        f"{task.id} under {mutation}: {[r.name for r in g.results if not r.passed]}"


@pytest.mark.parametrize("mutation", MUTATION_NAMES)
def test_server_contract_is_byte_identical(mutation, fresh_db):
    """Actions, input names and product/order hrefs must survive untouched."""
    with TestClient(create_app(fresh_db)) as clean, \
            TestClient(mutate_app(create_app(fresh_db), (mutation,))) as muted:
        for c in (clean, muted):
            login(c, *ALICE)
            add(c, "SHIRT-BLU", 1)          # give /cart and /checkout something to show
        for path in ("/", "/product/SHIRT-BLU", "/cart", "/checkout", "/orders"):
            assert _contract(clean.get(path).text) == _contract(muted.get(path).text), \
                f"{mutation} changed the server contract on {path}"


@pytest.mark.parametrize("mutation", MUTATION_NAMES)
def test_mutation_changes_the_surface(mutation, fresh_db):
    """No mutation is a silent no-op: it visibly alters at least one page."""
    with TestClient(create_app(fresh_db)) as clean, \
            TestClient(mutate_app(create_app(fresh_db), (mutation,))) as muted:
        for c in (clean, muted):
            login(c, *ALICE)
            add(c, "SHIRT-BLU", 1)
        pages = ("/", "/product/SHIRT-BLU", "/cart", "/checkout", "/orders")
        assert any(clean.get(p).text != muted.get(p).text for p in pages), \
            f"{mutation} changed nothing an agent can see"


def test_unknown_mutation_is_rejected(fresh_db):
    with pytest.raises(ValueError):
        mutate_app(create_app(fresh_db), ("no_such_mutation",))


def test_default_app_serves_mutations_and_faults_from_env(fresh_db, monkeypatch):
    """GT_MUTATIONS / GT_FAULTS let you browse a perturbed shop on localhost."""
    from app.main import default_app

    monkeypatch.setenv("GT_DB_PATH", str(fresh_db))
    monkeypatch.setenv("GT_MUTATIONS", "visual_noise")
    monkeypatch.setenv("GT_FAULTS", "error_on_checkout")
    with TestClient(default_app()) as c:
        login(c, *ALICE)
        assert "MEGA SALE" in c.get("/").text                    # mutation applied
        add(c, "SHIRT-BLU")
        r = c.post("/checkout", data={"ship_address": "x"}, follow_redirects=False)
        assert r.status_code == 500                               # fault applied


def test_mutations_compose(fresh_db):
    """All six at once still serves pages and keeps the contract on the storefront."""
    with TestClient(create_app(fresh_db)) as clean, \
            TestClient(mutate_app(create_app(fresh_db), MUTATION_NAMES)) as muted:
        login(clean, *ALICE)
        login(muted, *ALICE)
        assert _contract(clean.get("/").text) == _contract(muted.get("/").text)
        assert clean.get("/").text != muted.get("/").text
