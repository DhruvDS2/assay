"""
The shop — a deliberately small, server-rendered store that agents are tested against.

Every page is plain HTML with real forms and no JavaScript, so what the agent
sees is exactly what the server sent, and every action is a form POST that
lands in SQLite, where the grader can check it.

Run it:
    uvicorn app.main:default_app --factory --reload
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException
from starlette.middleware.sessions import SessionMiddleware

from app.db import connect, init_db, verify_password

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
TEMPLATES.env.filters["money"] = lambda cents: f"${cents / 100:,.2f}"


class LoginRequired(Exception):
    """Raised by require_user; turned into a redirect to /login."""


def create_app(db_path: str | Path, secret_key: str = "dev-only-secret") -> FastAPI:
    """Build a shop bound to one database file. One case = one app = one DB."""
    app = FastAPI(title="groundtruth shop", docs_url=None, redoc_url=None)
    app.add_middleware(SessionMiddleware, secret_key=secret_key)
    app.state.db_path = str(db_path)

    # ------------------------------------------------------------ plumbing

    def get_db():
        conn = connect(app.state.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def current_user(request: Request, db: sqlite3.Connection):
        uid = request.session.get("user_id")
        if uid is None:
            return None
        return db.execute(
            "SELECT id, email, name FROM users WHERE id = ?", (uid,)
        ).fetchone()

    def require_user(request: Request, db: sqlite3.Connection = Depends(get_db)):
        user = current_user(request, db)
        if user is None:
            raise LoginRequired()
        return user

    def render(request, db, user, template, status_code=200, **ctx):
        cart_count = 0
        if user is not None:
            cart_count = db.execute(
                "SELECT COALESCE(SUM(qty), 0) FROM cart_items WHERE user_id = ?",
                (user["id"],),
            ).fetchone()[0]
        return TEMPLATES.TemplateResponse(
            request,
            template,
            {"user": user, "cart_count": cart_count, **ctx},
            status_code=status_code,
        )

    def product_by_sku(db, sku: str):
        row = db.execute(
            "SELECT id, sku, name, price_cents, stock FROM products WHERE sku = ?",
            (sku,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "Product not found.")
        return row

    def cart_items(db, user_id: int):
        return db.execute(
            """
            SELECT p.id AS product_id, p.sku, p.name, p.price_cents, p.stock, c.qty
            FROM cart_items c JOIN products p ON p.id = c.product_id
            WHERE c.user_id = ?
            ORDER BY p.name
            """,
            (user_id,),
        ).fetchall()

    def own_order(db, user, order_id: int):
        row = db.execute(
            "SELECT * FROM orders WHERE id = ? AND user_id = ?", (order_id, user["id"])
        ).fetchone()
        if row is None:
            raise HTTPException(404, "Order not found.")
        return row

    @app.exception_handler(LoginRequired)
    def _to_login(request: Request, exc: LoginRequired):
        return RedirectResponse("/login", status_code=303)

    @app.exception_handler(HTTPException)
    def _error_page(request: Request, exc: HTTPException):
        return TEMPLATES.TemplateResponse(
            request,
            "error.html",
            {"user": None, "cart_count": 0, "status": exc.status_code, "detail": exc.detail},
            status_code=exc.status_code,
        )

    # ------------------------------------------------------------ auth

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request, db=Depends(get_db)):
        return render(request, db, None, "login.html")

    @app.post("/login")
    def login(
        request: Request,
        email: str = Form(...),
        password: str = Form(...),
        db=Depends(get_db),
    ):
        row = db.execute(
            "SELECT id, password_hash FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return render(
                request, db, None, "login.html",
                status_code=401, error="Incorrect email or password.", email=email,
            )
        request.session.clear()
        request.session["user_id"] = row["id"]
        return RedirectResponse("/", status_code=303)

    @app.post("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    # ------------------------------------------------------------ browsing

    @app.get("/", response_class=HTMLResponse)
    def products(request: Request, db=Depends(get_db), user=Depends(require_user)):
        rows = db.execute(
            "SELECT sku, name, price_cents, stock FROM products ORDER BY name"
        ).fetchall()
        return render(request, db, user, "products.html", products=rows)

    @app.get("/product/{sku}", response_class=HTMLResponse)
    def product(sku: str, request: Request, db=Depends(get_db), user=Depends(require_user)):
        return render(request, db, user, "product.html", product=product_by_sku(db, sku))

    # ------------------------------------------------------------ cart

    @app.get("/cart", response_class=HTMLResponse)
    def cart(request: Request, db=Depends(get_db), user=Depends(require_user)):
        items = cart_items(db, user["id"])
        total = sum(i["qty"] * i["price_cents"] for i in items)
        return render(request, db, user, "cart.html", items=items, total=total)

    @app.post("/cart/add")
    def cart_add(
        sku: str = Form(...),
        qty: int = Form(1),
        db=Depends(get_db),
        user=Depends(require_user),
    ):
        if qty < 1:
            raise HTTPException(400, "Quantity must be at least 1.")
        p = product_by_sku(db, sku)
        with db:
            db.execute(
                """
                INSERT INTO cart_items (user_id, product_id, qty) VALUES (?, ?, ?)
                ON CONFLICT (user_id, product_id) DO UPDATE SET qty = qty + excluded.qty
                """,
                (user["id"], p["id"], qty),
            )
        return RedirectResponse("/cart", status_code=303)

    @app.post("/cart/update")
    def cart_update(
        sku: str = Form(...),
        qty: int = Form(...),
        db=Depends(get_db),
        user=Depends(require_user),
    ):
        if qty < 0:
            raise HTTPException(400, "Quantity cannot be negative.")
        p = product_by_sku(db, sku)
        with db:
            if qty == 0:
                db.execute(
                    "DELETE FROM cart_items WHERE user_id = ? AND product_id = ?",
                    (user["id"], p["id"]),
                )
            else:
                db.execute(
                    "UPDATE cart_items SET qty = ? WHERE user_id = ? AND product_id = ?",
                    (qty, user["id"], p["id"]),
                )
        return RedirectResponse("/cart", status_code=303)

    @app.post("/cart/clear")
    def cart_clear(db=Depends(get_db), user=Depends(require_user)):
        with db:
            db.execute("DELETE FROM cart_items WHERE user_id = ?", (user["id"],))
        return RedirectResponse("/cart", status_code=303)

    # ------------------------------------------------------------ checkout

    @app.get("/checkout", response_class=HTMLResponse)
    def checkout_page(request: Request, db=Depends(get_db), user=Depends(require_user)):
        items = cart_items(db, user["id"])
        if not items:
            return RedirectResponse("/cart", status_code=303)
        total = sum(i["qty"] * i["price_cents"] for i in items)
        return render(request, db, user, "checkout.html", items=items, total=total)

    @app.post("/checkout")
    def place_order(
        ship_address: str = Form(""),
        db=Depends(get_db),
        user=Depends(require_user),
    ):
        address = ship_address.strip()
        if not address:
            raise HTTPException(400, "A shipping address is required.")
        items = cart_items(db, user["id"])
        if not items:
            raise HTTPException(400, "Your cart is empty.")
        for i in items:
            if i["qty"] > i["stock"]:
                raise HTTPException(409, f"Only {i['stock']} {i['name']} left in stock.")

        total = sum(i["qty"] * i["price_cents"] for i in items)
        # One transaction: the order, its lines, the stock, and the cart
        # all change together or not at all.
        with db:
            order_id = db.execute(
                "INSERT INTO orders (user_id, status, total_cents, ship_address) "
                "VALUES (?, 'placed', ?, ?)",
                (user["id"], total, address),
            ).lastrowid
            db.executemany(
                "INSERT INTO order_items (order_id, product_id, qty, price_cents) "
                "VALUES (?, ?, ?, ?)",
                [(order_id, i["product_id"], i["qty"], i["price_cents"]) for i in items],
            )
            db.executemany(
                "UPDATE products SET stock = stock - ? WHERE id = ?",
                [(i["qty"], i["product_id"]) for i in items],
            )
            db.execute("DELETE FROM cart_items WHERE user_id = ?", (user["id"],))
        return RedirectResponse(f"/orders/{order_id}", status_code=303)

    # ------------------------------------------------------------ orders

    @app.get("/orders", response_class=HTMLResponse)
    def orders(request: Request, db=Depends(get_db), user=Depends(require_user)):
        rows = db.execute(
            """
            SELECT o.id, o.status, o.total_cents, o.created_at,
                   (SELECT SUM(qty) FROM order_items WHERE order_id = o.id) AS n_items
            FROM orders o
            WHERE o.user_id = ?
            ORDER BY o.created_at DESC, o.id DESC
            """,
            (user["id"],),
        ).fetchall()
        return render(request, db, user, "orders.html", orders=rows)

    @app.get("/orders/{order_id}", response_class=HTMLResponse)
    def order_detail(
        order_id: int, request: Request, db=Depends(get_db), user=Depends(require_user)
    ):
        order = own_order(db, user, order_id)
        lines = db.execute(
            """
            SELECT p.sku, p.name, oi.qty, oi.price_cents
            FROM order_items oi JOIN products p ON p.id = oi.product_id
            WHERE oi.order_id = ?
            ORDER BY p.name
            """,
            (order_id,),
        ).fetchall()
        return render(request, db, user, "order.html", order=order, lines=lines)

    @app.post("/orders/{order_id}/cancel")
    def cancel_order(order_id: int, db=Depends(get_db), user=Depends(require_user)):
        order = own_order(db, user, order_id)
        if order["status"] != "placed":
            raise HTTPException(409, "Only placed orders can be cancelled.")
        with db:
            db.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (order_id,))
            db.execute(
                """
                UPDATE products SET stock = stock + (
                    SELECT qty FROM order_items
                    WHERE order_id = ? AND product_id = products.id
                )
                WHERE id IN (SELECT product_id FROM order_items WHERE order_id = ?)
                """,
                (order_id, order_id),
            )
        return RedirectResponse(f"/orders/{order_id}", status_code=303)

    return app


def _env_list(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def default_app() -> FastAPI:
    """
    Factory for uvicorn. Builds data/app.db on first run.

    For hands-on inspection, `GT_MUTATIONS` and `GT_FAULTS` (comma-separated names)
    serve the perturbed shop so you can *see* in a browser what an agent sees under
    a condition — e.g. `GT_MUTATIONS=visual_noise,rename_controls uvicorn ...`.
    Lazy imports keep app.main free of a cycle with app.mutations/app.faults.
    """
    path = Path(os.environ.get("GT_DB_PATH", "data/app.db"))
    if not path.exists():
        init_db(path)
    app = create_app(path, os.environ.get("GT_SECRET_KEY", "dev-only-secret"))

    mutations = _env_list("GT_MUTATIONS")
    if mutations:
        from app.mutations import mutate_app
        app = mutate_app(app, mutations)

    faults = _env_list("GT_FAULTS")
    if faults:
        from app.faults import inject_faults
        app = inject_faults(app, faults)

    return app
