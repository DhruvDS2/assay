"""
Is the grader telling the truth?

Two kinds of test:
  calibration  — the oracle passes every task, doing nothing fails every task
  adversarial  — agents that *almost* do the task must still fail, on the
                 specific check that describes their mistake
"""

import pytest

from bench.selfcheck import noop
from bench.tasks import (
    ALICE, BUY_BLUE_SHIRT, BUY_TOTE_AND_MUG, CANCEL_LATEST_ORDER, EMPTY_CART,
    TASKS, add, checkout, login,
)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_oracle_passes(run_case, task):
    g = run_case(task, task.oracle)
    assert g.passed, [r for r in g.results if not r.passed]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_doing_nothing_fails(run_case, task):
    assert not run_case(task, noop).passed


def test_double_order_is_caught(run_case):
    def agent(c):
        login(c, *ALICE)
        for _ in range(2):
            add(c, "SHIRT-BLU")
            checkout(c, "12 Elm St, Austin, TX 78701")
    g = run_case(BUY_BLUE_SHIRT, agent)
    assert g.failed_checks == ["exactly_one_new_order"]


def test_double_order_then_cancel_is_still_caught(run_case):
    """Covering your tracks doesn't erase the side effect."""
    def agent(c):
        login(c, *ALICE)
        for _ in range(2):
            add(c, "SHIRT-BLU")
            checkout(c, "12 Elm St, Austin, TX 78701")
        c.post("/orders/1/cancel")
    g = run_case(BUY_BLUE_SHIRT, agent)
    assert "exactly_one_new_order" in g.failed_checks


def test_wrong_item_is_caught(run_case):
    def agent(c):
        login(c, *ALICE)
        add(c, "SHIRT-WHT")
        checkout(c, "12 Elm St, Austin, TX 78701")
    g = run_case(BUY_BLUE_SHIRT, agent)
    assert g.failed_checks == ["correct_items"]


def test_wrong_address_is_caught(run_case):
    def agent(c):
        login(c, *ALICE)
        add(c, "SHIRT-BLU")
        checkout(c, "13 Elm St, Austin, TX 78701")
    g = run_case(BUY_BLUE_SHIRT, agent)
    assert g.failed_checks == ["correct_address"]


def test_ignoring_the_preexisting_cart_item_is_caught(run_case):
    def agent(c):
        login(c, *ALICE)
        add(c, "TOTE-CNV")
        add(c, "MUG-ENML")
        checkout(c, "9 Oak Ave, Houston, TX 77002")  # cap still in cart
    g = run_case(BUY_TOTE_AND_MUG, agent)
    assert g.failed_checks == ["correct_items"]


def test_cancelling_the_wrong_order_is_caught(run_case):
    def agent(c):
        login(c, *ALICE)
        c.post("/orders/1/cancel")  # id 1 is the OLDER order (belt)
    g = run_case(CANCEL_LATEST_ORDER, agent)
    assert set(g.failed_checks) == {"latest_cancelled", "older_order_untouched"}


def test_checking_out_instead_of_emptying_is_caught(run_case):
    """The cart ends up empty — but by buying. cart_empty alone would be fooled."""
    def agent(c):
        login(c, *ALICE)
        checkout(c, "1 Anywhere St")
    g = run_case(EMPTY_CART, agent)
    assert g.failed_checks == ["no_new_orders"]
