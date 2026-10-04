"""
B3 receipt grader: it reads the order page as a structured receipt, so it catches
what a keyword scrape can't — a wrong item, and crucially a wrong *quantity*
("Wool Socks 1" still contains the words "wool socks").

Correct blue-shirt buy passes; the look-alike White shirt fails (wrong item); a
single pair of socks fails the x3 task (wrong quantity). Cases are built the same
way as tests/test_graders.py: prepare a fresh case, drive the real forms, grade.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from bench.env import prepare_case
from bench.graders import render_state
from bench.receipt_grader import ReceiptGrader, parse_receipt
from bench.tasks import BUY_BLUE_SHIRT, BUY_SOCKS_X3, BUY_TOTE_AND_MUG, add, checkout, login


def _buy(task, seed, tmp_path, label, skus, address):
    """Prepare a fresh case for `task` and buy `skus` ({sku: qty}) in one order."""
    case = prepare_case(task, tmp_path / label, seed)
    with TestClient(create_app(case.db_path)) as c:
        login(c, "alice@example.com", "password123")
        for sku, qty in skus.items():
            add(c, sku, qty)
        checkout(c, address)
    return case


def test_parse_receipt_reads_items_qty_and_address(seed, tmp_path):
    case = _buy(BUY_BLUE_SHIRT, seed, tmp_path, "blue", {"SHIRT-BLU": 1}, "12 Elm St, Austin, TX 78701")
    r = parse_receipt(render_state(case)["/orders/latest"])
    assert r.status == "placed"
    assert r.items == {"blue oxford shirt": 1}
    assert "12 elm st" in r.ship_address


def test_receipt_passes_correct_blue_shirt(seed, tmp_path):
    case = _buy(BUY_BLUE_SHIRT, seed, tmp_path, "blue", {"SHIRT-BLU": 1}, "12 Elm St, Austin, TX 78701")
    v = ReceiptGrader().judge(case, render_state(case))
    assert v.grader == "receipt" and v.passed, v.detail


def test_receipt_fails_lookalike_white_shirt(seed, tmp_path):
    # Same task (buy the BLUE shirt), but the agent bought the White look-alike.
    case = _buy(BUY_BLUE_SHIRT, seed, tmp_path, "white", {"SHIRT-WHT": 1}, "12 Elm St, Austin, TX 78701")
    v = ReceiptGrader().judge(case, render_state(case))
    assert not v.passed   # receipt reads the item name, not just "an order was placed"


def test_receipt_fails_wrong_quantity_socks(seed, tmp_path):
    # Task wants 3 pairs; the agent bought 1. A keyword scrape for "wool socks"
    # would still pass — the parsed quantity is what fails it.
    case = _buy(BUY_SOCKS_X3, seed, tmp_path, "socks1", {"SOCK-WOOL": 1}, "400 Main St, Dallas, TX 75201")
    v = ReceiptGrader().judge(case, render_state(case))
    assert not v.passed

    # And the correct x3 order passes, proving the quantity check is real.
    good = _buy(BUY_SOCKS_X3, seed, tmp_path, "socks3", {"SOCK-WOOL": 3}, "400 Main St, Dallas, TX 75201")
    assert ReceiptGrader().judge(good, render_state(good)).passed


def test_receipt_reads_a_two_item_order(seed, tmp_path):
    # The cart starts with a Black Cap (task setup); the oracle path removes it.
    case = prepare_case(BUY_TOTE_AND_MUG, tmp_path / "tm", seed)
    with TestClient(create_app(case.db_path)) as c:
        login(c, "alice@example.com", "password123")
        c.post("/cart/update", data={"sku": "CAP-BLK", "qty": 0})
        add(c, "TOTE-CNV")
        add(c, "MUG-ENML")
        checkout(c, "9 Oak Ave, Houston, TX 77002")
    v = ReceiptGrader().judge(case, render_state(case))
    assert v.passed, v.detail
