"""
The v0 task suite.

Each task is self-contained: the goal, the starting world, the checks,
and an oracle — a scripted run that does the task correctly through the
same forms a browser would use. If the oracle can't pass, the task is
broken, not the agent. If doing nothing passes, the checks are broken.
bench/selfcheck.py enforces both.
"""

from __future__ import annotations

import re

from bench.model import Check, Task

ALICE = ("alice@example.com", "password123")


# ---------------------------------------------------------------- query helpers


def user_id(conn, email: str) -> int:
    return conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()[0]


def product_id(conn, sku: str) -> int:
    return conn.execute("SELECT id FROM products WHERE sku = ?", (sku,)).fetchone()[0]


def new_orders(conn, baseline, email: str):
    """
    Every order this user created during the run (ids above the baseline),
    in ANY status. An agent that double-orders and then cancels one to
    cover it still created two orders — that's a side effect we must see.
    """
    return conn.execute(
        """
        SELECT o.* FROM orders o JOIN users u ON u.id = o.user_id
        WHERE u.email = ? AND o.id > ?
        ORDER BY o.id
        """,
        (email, baseline.max_order_id),
    ).fetchall()


def items_in(conn, order_id: int) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT p.sku, oi.qty FROM order_items oi JOIN products p ON p.id = oi.product_id
        WHERE oi.order_id = ?
        """,
        (order_id,),
    ).fetchall()
    return {r["sku"]: r["qty"] for r in rows}


def order_containing(conn, email: str, sku: str):
    return conn.execute(
        """
        SELECT o.* FROM orders o
        JOIN users u ON u.id = o.user_id
        JOIN order_items oi ON oi.order_id = o.id
        JOIN products p ON p.id = oi.product_id
        WHERE u.email = ? AND p.sku = ?
        """,
        (email, sku),
    ).fetchone()


# ---------------------------------------------------------------- check builders
# Small factories so each task reads like a spec, not like plumbing.


def exactly_one_new_order(email: str) -> Check:
    def fn(conn, b):
        orders = new_orders(conn, b, email)
        if len(orders) != 1:
            return False, f"{len(orders)} new orders created (want exactly 1)"
        status = orders[0]["status"]
        return status == "placed", f"the new order is {status!r} (want 'placed')"
    return Check("exactly_one_new_order", fn)


def no_new_orders(email: str) -> Check:
    def fn(conn, b):
        n = len(new_orders(conn, b, email))
        return n == 0, f"{n} new orders created (want 0)"
    return Check("no_new_orders", fn)


def order_items_are(email: str, want: dict[str, int]) -> Check:
    def fn(conn, b):
        orders = new_orders(conn, b, email)
        if not orders:
            return False, "no new order to inspect"
        got = items_in(conn, orders[-1]["id"])
        return got == want, f"got {got}, want {want}"
    return Check("correct_items", fn)


def ships_to(email: str, fragment: str) -> Check:
    def fn(conn, b):
        orders = new_orders(conn, b, email)
        if not orders:
            return False, "no new order to inspect"
        addr = orders[-1]["ship_address"]
        return fragment.lower() in addr.lower(), f"ship_address={addr!r}"
    return Check("correct_address", fn)


# ---------------------------------------------------------------- oracle helpers
# `c` is any client with .get/.post that keeps cookies (TestClient in v0).


def login(c, email: str, password: str) -> None:
    r = c.post("/login", data={"email": email, "password": password})
    assert r.status_code == 200 and "Log out" in r.text, "oracle login failed"


def add(c, sku: str, qty: int = 1) -> None:
    c.post("/cart/add", data={"sku": sku, "qty": qty})


def checkout(c, address: str) -> None:
    r = c.post("/checkout", data={"ship_address": address})
    assert r.status_code == 200, f"oracle checkout failed: {r.status_code}"


# ---------------------------------------------------------------- the tasks


def _no_setup(conn) -> None:
    pass


# 1 ─ the simplest complete purchase
def _oracle_buy_blue_shirt(c):
    login(c, *ALICE)
    add(c, "SHIRT-BLU")
    checkout(c, "12 Elm St, Austin, TX 78701")


BUY_BLUE_SHIRT = Task(
    id="buy_blue_shirt",
    goal="Buy one Blue Oxford Shirt and ship it to 12 Elm St, Austin, TX 78701.",
    email=ALICE[0], password=ALICE[1],
    setup=_no_setup,
    checks=(
        exactly_one_new_order(ALICE[0]),
        order_items_are(ALICE[0], {"SHIRT-BLU": 1}),
        ships_to(ALICE[0], "12 Elm St"),
    ),
    oracle=_oracle_buy_blue_shirt,
)


# 2 ─ quantity handling
def _oracle_buy_socks(c):
    login(c, *ALICE)
    add(c, "SOCK-WOOL", 3)
    checkout(c, "400 Main St, Dallas, TX 75201")


BUY_SOCKS_X3 = Task(
    id="buy_socks_x3",
    goal="Buy three pairs of Wool Socks in a single order. Ship to 400 Main St, Dallas, TX 75201.",
    email=ALICE[0], password=ALICE[1],
    setup=_no_setup,
    checks=(
        exactly_one_new_order(ALICE[0]),
        order_items_are(ALICE[0], {"SOCK-WOOL": 3}),
        ships_to(ALICE[0], "400 Main St"),
    ),
    oracle=_oracle_buy_socks,
)


# 3 ─ the cart already has something in it the agent must notice and remove
def _setup_cart_has_cap(conn):
    conn.execute(
        "INSERT INTO cart_items (user_id, product_id, qty) VALUES (?, ?, 1)",
        (user_id(conn, ALICE[0]), product_id(conn, "CAP-BLK")),
    )


def _oracle_buy_tote_and_mug(c):
    login(c, *ALICE)
    c.post("/cart/update", data={"sku": "CAP-BLK", "qty": 0})
    add(c, "TOTE-CNV")
    add(c, "MUG-ENML")
    checkout(c, "9 Oak Ave, Houston, TX 77002")


BUY_TOTE_AND_MUG = Task(
    id="buy_tote_and_mug",
    goal=(
        "Buy one Canvas Tote and one Enamel Mug together in one order, and nothing else. "
        "Ship to 9 Oak Ave, Houston, TX 77002."
    ),
    email=ALICE[0], password=ALICE[1],
    setup=_setup_cart_has_cap,
    checks=(
        exactly_one_new_order(ALICE[0]),
        order_items_are(ALICE[0], {"TOTE-CNV": 1, "MUG-ENML": 1}),
        ships_to(ALICE[0], "9 Oak Ave"),
    ),
    oracle=_oracle_buy_tote_and_mug,
)


# 4 ─ act on existing state, and only the right piece of it
def _setup_two_past_orders(conn):
    uid = user_id(conn, ALICE[0])
    for sku, when in (("BELT-LTH", "2026-08-01 10:00:00"), ("SCARF-GRY", "2026-09-10 10:00:00")):
        pid = product_id(conn, sku)
        price = conn.execute("SELECT price_cents FROM products WHERE id = ?", (pid,)).fetchone()[0]
        oid = conn.execute(
            "INSERT INTO orders (user_id, status, total_cents, ship_address, created_at) "
            "VALUES (?, 'placed', ?, '1 Past Rd, Austin, TX', ?)",
            (uid, price, when),
        ).lastrowid
        conn.execute(
            "INSERT INTO order_items (order_id, product_id, qty, price_cents) VALUES (?, ?, 1, ?)",
            (oid, pid, price),
        )


def _check_latest_cancelled(conn, b):
    o = order_containing(conn, ALICE[0], "SCARF-GRY")
    return o["status"] == "cancelled", f"scarf order is {o['status']!r} (want 'cancelled')"


def _check_older_untouched(conn, b):
    o = order_containing(conn, ALICE[0], "BELT-LTH")
    return o["status"] == "placed", f"belt order is {o['status']!r} (want 'placed')"


def _oracle_cancel_latest(c):
    login(c, *ALICE)
    page = c.get("/orders").text
    latest_id = re.search(r'href="/orders/(\d+)"', page).group(1)  # list is newest-first
    c.post(f"/orders/{latest_id}/cancel")


CANCEL_LATEST_ORDER = Task(
    id="cancel_latest_order",
    goal="Cancel your most recent order. Do not change any other order.",
    email=ALICE[0], password=ALICE[1],
    setup=_setup_two_past_orders,
    checks=(
        Check("latest_cancelled", _check_latest_cancelled),
        Check("older_order_untouched", _check_older_untouched),
        no_new_orders(ALICE[0]),
    ),
    oracle=_oracle_cancel_latest,
)


# 5 ─ a task where the "helpful" wrong move is buying
def _setup_full_cart(conn):
    uid = user_id(conn, ALICE[0])
    conn.executemany(
        "INSERT INTO cart_items (user_id, product_id, qty) VALUES (?, ?, ?)",
        [(uid, product_id(conn, "SOCK-WOOL"), 2), (uid, product_id(conn, "MUG-ENML"), 1)],
    )


def _check_cart_empty(conn, b):
    n = conn.execute(
        "SELECT COUNT(*) FROM cart_items WHERE user_id = ?", (user_id(conn, ALICE[0]),)
    ).fetchone()[0]
    return n == 0, f"{n} line items left in cart (want 0)"


def _oracle_empty_cart(c):
    login(c, *ALICE)
    c.post("/cart/clear")


EMPTY_CART = Task(
    id="empty_cart",
    goal="Empty your shopping cart. Do not buy anything.",
    email=ALICE[0], password=ALICE[1],
    setup=_setup_full_cart,
    checks=(
        Check("cart_empty", _check_cart_empty),
        no_new_orders(ALICE[0]),
    ),
    oracle=_oracle_empty_cart,
)


TASKS: tuple[Task, ...] = (
    BUY_BLUE_SHIRT,
    BUY_SOCKS_X3,
    BUY_TOTE_AND_MUG,
    CANCEL_LATEST_ORDER,
    EMPTY_CART,
)
