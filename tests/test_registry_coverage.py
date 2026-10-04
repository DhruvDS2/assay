"""
Registry coverage: a task must not be added without the checks/models that grade it.

Three registries key behavior off `task.id`, and a task silently missing from one
is exactly the kind of gap that passes CI while quietly measuring nothing:

  - `_DOM_CHECKS` (bench/graders.py)        — the DOM grader covers EVERY task.
  - `_RECEIPT_CHECKS` (bench/receipt_grader.py) — purchase tasks only (it abstains
                                                  elsewhere by design).
  - `HARDENED_MODELS` (bench/hardened.py)   — purchase tasks only (a checkout fault
                                              can only bite a purchase).

These tests pin the intended coverage so adding a 6th task — or a new purchase
task — fails loudly here until its registries are updated. They also catch a
stale or typo'd key that names no real task.
"""

from __future__ import annotations

from bench.graders import _DOM_CHECKS
from bench.hardened import HARDENED_MODELS
from bench.receipt_grader import _RECEIPT_CHECKS
from bench.tasks import TASKS

ALL_TASK_IDS = {t.id for t in TASKS}

# The purchase tasks — the ones a checkout fault can bite and the ones with a
# structured receipt. If you add a purchase task, add it here (that's the point:
# a conscious classification, enforced by the asserts below).
PURCHASE_TASK_IDS = {"buy_blue_shirt", "buy_socks_x3", "buy_tote_and_mug"}


def test_purchase_ids_are_real_tasks():
    """Guard the test's own source of truth against drift."""
    assert PURCHASE_TASK_IDS <= ALL_TASK_IDS


def test_dom_checks_cover_every_task():
    """The DOM grader is the general one: no task may lack a DOM check."""
    assert set(_DOM_CHECKS) == ALL_TASK_IDS, (
        f"DOM check coverage drifted: missing {ALL_TASK_IDS - set(_DOM_CHECKS)}, "
        f"stale {set(_DOM_CHECKS) - ALL_TASK_IDS}"
    )


def test_receipt_checks_cover_exactly_the_purchase_tasks():
    assert set(_RECEIPT_CHECKS) == PURCHASE_TASK_IDS, (
        f"receipt coverage drifted: missing {PURCHASE_TASK_IDS - set(_RECEIPT_CHECKS)}, "
        f"unexpected {set(_RECEIPT_CHECKS) - PURCHASE_TASK_IDS}"
    )


def test_hardened_models_cover_exactly_the_purchase_tasks():
    assert set(HARDENED_MODELS) == PURCHASE_TASK_IDS, (
        f"hardened-model coverage drifted: missing {PURCHASE_TASK_IDS - set(HARDENED_MODELS)}, "
        f"unexpected {set(HARDENED_MODELS) - PURCHASE_TASK_IDS}"
    )


def test_no_registry_names_a_nonexistent_task():
    """Every key in every registry must name a real task — catches typos/stale keys."""
    for name, reg in (("_DOM_CHECKS", _DOM_CHECKS),
                      ("_RECEIPT_CHECKS", _RECEIPT_CHECKS),
                      ("HARDENED_MODELS", HARDENED_MODELS)):
        stale = set(reg) - ALL_TASK_IDS
        assert not stale, f"{name} has keys that name no task: {stale}"
