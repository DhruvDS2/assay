"""The shop itself behaves correctly. If these fail, no grade can be trusted."""

from app.db import connect


def login(client, email="alice@example.com", password="password123"):
    return client.post("/login", data={"email": email, "password": password})


def test_pages_require_login(shop):
    client, _ = shop
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"


def test_bad_password_is_rejected(shop):
    client, _ = shop
    r = login(client, password="wrong")
    assert r.status_code == 401 and "Incorrect email or password" in r.text


def test_checkout_writes_order_decrements_stock_clears_cart(shop):
    client, db = shop
    login(client)
    client.post("/cart/add", data={"sku": "SOCK-WOOL", "qty": 3})
    r = client.post("/checkout", data={"ship_address": "400 Main St"})
    assert r.status_code == 200 and "Order #" in r.text

    conn = connect(db)
    assert conn.execute("SELECT stock FROM products WHERE sku='SOCK-WOOL'").fetchone()[0] == 47
    assert conn.execute("SELECT COUNT(*) FROM cart_items").fetchone()[0] == 0
    assert conn.execute("SELECT total_cents FROM orders").fetchone()[0] == 3 * 1400
    conn.close()


def test_cancel_restocks(shop):
    client, db = shop
    login(client)
    client.post("/cart/add", data={"sku": "BELT-LTH", "qty": 2})
    client.post("/checkout", data={"ship_address": "1 Rd"})
    client.post("/orders/1/cancel")

    conn = connect(db)
    assert conn.execute("SELECT status FROM orders WHERE id=1").fetchone()[0] == "cancelled"
    assert conn.execute("SELECT stock FROM products WHERE sku='BELT-LTH'").fetchone()[0] == 8
    conn.close()


def test_cannot_cancel_twice(shop):
    client, _ = shop
    login(client)
    client.post("/cart/add", data={"sku": "CAP-BLK"})
    client.post("/checkout", data={"ship_address": "1 Rd"})
    client.post("/orders/1/cancel")
    assert client.post("/orders/1/cancel").status_code == 409


def test_cannot_see_another_users_order(shop):
    client, _ = shop
    login(client)
    client.post("/cart/add", data={"sku": "CAP-BLK"})
    client.post("/checkout", data={"ship_address": "1 Rd"})
    client.post("/logout")
    login(client, "bob@example.com", "hunter22")
    assert client.get("/orders/1").status_code == 404


def test_empty_cart_checkout_is_rejected(shop):
    client, _ = shop
    login(client)
    assert client.post("/checkout", data={"ship_address": "1 Rd"}).status_code == 400


def test_overselling_is_rejected(shop):
    client, _ = shop
    login(client)
    client.post("/cart/add", data={"sku": "BELT-LTH", "qty": 9})  # only 8 in stock
    assert client.post("/checkout", data={"ship_address": "1 Rd"}).status_code == 409
