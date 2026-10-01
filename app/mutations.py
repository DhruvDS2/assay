"""
UI mutations — the six ways we deliberately break the shop's *surface* to see
what breaks the agent.

The contract every mutation keeps, so a mutated case is still gradable:

- **It never touches what the server reads.** Form `action=`s, input `name=`s,
  the SKUs inside hidden fields, and every `/orders/<id>` / `/product/<sku>`
  href stay byte-for-byte identical. A correct action still lands in SQLite
  exactly as before, so the grader's SQL is unaffected and the DB stays truth.
- **It only rewrites presentation** — the visible text the agent reads, the
  order rows appear in, decoy links, nav structure, cosmetic noise. The thing
  an agent must *see through*, never the thing the server acts on.

Because the oracle drives the shop by its endpoints (and only ever reads
`/orders` hrefs, which we preserve), every task stays achievable under every
mutation — that's the point: the oracle proves the task is still doable while
the mutation measures whether the *agent* can still do it.

Mutations are applied as an outgoing-HTML transform via middleware, so the app
and its templates never learn they exist. `clean` + these six = the roadmap's
seven headline conditions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.main import create_app

Transform = Callable[[str], str]


@dataclass(frozen=True)
class Mutation:
    """One named, presentation-only perturbation of the shop's HTML."""

    name: str
    description: str
    transform: Transform


# --------------------------------------------------------------------- helpers

# A <tbody> whose rows link to product pages — i.e. the storefront listing, not
# the cart/checkout/orders tables (which have no /product/ links).
_PRODUCT_TBODY = re.compile(r"<tbody>(?P<rows>.*?)</tbody>", re.DOTALL)
_TR = re.compile(r"<tr>.*?</tr>", re.DOTALL)


def _rename_text(html: str, pairs: dict[str, str]) -> str:
    """Swap visible inner text `>old<` -> `>new<`, leaving attributes alone."""
    for old, new in pairs.items():
        html = html.replace(f">{old}<", f">{new}<")
    return html


# ------------------------------------------------------------------ the six

def _rename_controls(html: str) -> str:
    """
    Rename every actionable button/link the tasks depend on. The form each one
    submits is untouched — only the words on it change, so an agent that keys off
    exact labels ("click 'Add to cart'") misfires while the endpoint is the same.
    """
    html = _rename_text(html, {
        "Add to cart": "Add to bag",
        "Update": "Save changes",
        "Empty cart": "Clear basket",
        "Place order": "Complete purchase",
        "Cancel order": "Void this order",
        "Proceed to checkout</a": "Go to payment</a",
    })
    # Remove {{name}} -> Take out {{name}} (prefix match keeps the product name).
    return html.replace(">Remove ", ">Take out ")


def _reorder_products(html: str) -> str:
    """
    Reverse the storefront listing. Position-based selection ("buy the first
    shirt") now lands on the wrong item, sharpening every look-alike trap — the
    White vs Blue Oxford Shirt especially. SKUs and hrefs ride along unchanged.
    """
    def flip(m: re.Match) -> str:
        rows = m.group("rows")
        if "/product/" not in rows:          # not the storefront table — leave it
            return m.group(0)
        trs = _TR.findall(rows)
        return "<tbody>" + "".join(reversed(trs)) + "</tbody>"

    return _PRODUCT_TBODY.sub(flip, html)


def _inject_decoys(html: str) -> str:
    """
    Add plausible distractors that lead the wrong way. A prominent "featured"
    call-to-action points at the look-alike White shirt, and a "Deals" nav link
    goes nowhere useful. The real controls all remain; a robust agent ignores
    the bait, a credulous one follows it.
    """
    html = html.replace(
        "<h1>Products</h1>",
        '<h1>Products</h1>\n'
        '<p><a href="/product/SHIRT-WHT" '
        'style="font-size:1.2rem;font-weight:bold">'
        '⭐ Featured deal — Oxford Shirt, buy now!</a></p>',
    )
    # A tempting nav entry that just loops back home.
    html = html.replace(
        '<a href="/orders">Orders</a>',
        '<a href="/orders">Orders</a>\n      <a href="/">Deals</a>',
    )
    return html


def _relabel_fields(html: str) -> str:
    """
    Rename the labels and column headers the agent reads to locate a field,
    without touching a single input `name`/`id`. "Shipping address" becomes
    "Delivery destination", "Quantity" becomes "How many?", and so on — the
    form still posts the same keys.
    """
    html = _rename_text(html, {
        "Shipping address": "Delivery destination",
        "Quantity</th": "Qty</th",
        "Price</th": "Cost</th",
        "In stock</th": "Available</th",
        "Product</th": "Item</th",
    })
    # Product-page and cart quantity labels (text, not the <input name="qty">).
    html = html.replace(">Quantity<", ">How many?<")
    return html.replace(">Quantity for ", ">How many of ")


def _restructure_nav(html: str) -> str:
    """
    Bury the Cart and Orders links behind a collapsed <details> disclosure. Both
    hrefs still exist — the agent just has to realise navigation is now a menu it
    must open, not links sitting in plain sight.
    """
    nav = re.compile(
        r'(<a href="/cart">.*?</a>)\s*(<a href="/orders">Orders</a>)',
        re.DOTALL,
    )
    return nav.sub(
        r"<details><summary>Menu</summary>\1 \2</details>",
        html,
    )


def _visual_noise(html: str) -> str:
    """
    Wrap the page in marketing clutter: a sale banner and a (non-interactive)
    cookie notice. Pure decoration — no controls, no JavaScript, no dialogs —
    but plenty for a screenshot-reading agent to get lost in.
    """
    banner = (
        '<div style="background:#ffd54f;padding:.6rem;text-align:center">'
        '\U0001f389 MEGA SALE — up to 70% off selected items! Ends soon ⏰'
        '</div>\n'
        '<div style="background:#eee;padding:.4rem;font-size:.85rem">'
        'We value your privacy and use cookies to improve your experience. '
        'By browsing you agree to our policy.</div>\n'
    )
    return html.replace("<body>", "<body>\n" + banner, 1)


MUTATIONS: dict[str, Mutation] = {
    m.name: m
    for m in (
        Mutation("rename_controls",
                 "rename every task-critical button and link", _rename_controls),
        Mutation("reorder_products",
                 "reverse the storefront listing order", _reorder_products),
        Mutation("inject_decoys",
                 "add plausible distractor links that mislead", _inject_decoys),
        Mutation("relabel_fields",
                 "rename field labels and column headers", _relabel_fields),
        Mutation("restructure_nav",
                 "hide Cart/Orders behind a disclosure menu", _restructure_nav),
        Mutation("visual_noise",
                 "inject sale banner and cookie-notice clutter", _visual_noise),
    )
}

# Stable order — also the order the headline matrix reports them in.
MUTATION_NAMES: tuple[str, ...] = tuple(MUTATIONS)

# The seven headline conditions as plain data: clean, then one mutation each.
# `run.py` turns these into `Condition`s; keeping them here (not there) means the
# app layer owns what a condition *is* and `bench` never has to know the markup.
SEVEN_CONDITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("clean", ()),
    *((name, (name,)) for name in MUTATION_NAMES),
)


# ------------------------------------------------------------------ application

def apply(html: str, mutations: tuple[str, ...]) -> str:
    """Run `html` through each named mutation, in the given order."""
    for name in mutations:
        html = MUTATIONS[name].transform(html)
    return html


class _MutationMiddleware(BaseHTTPMiddleware):
    """Rewrite outgoing text/html bodies; pass redirects and JSON straight through."""

    def __init__(self, app, mutations: tuple[str, ...]) -> None:
        super().__init__(app)
        self._mutations = mutations

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if "text/html" not in response.headers.get("content-type", ""):
            return response

        body = b"".join([chunk async for chunk in response.body_iterator])
        html = apply(body.decode("utf-8"), self._mutations)

        # New body => let Response recompute content-length/content-type; carry
        # every other header (Set-Cookie included, which dict() would collapse).
        out = Response(content=html, status_code=response.status_code,
                       media_type="text/html")
        for key, value in response.headers.raw:
            if key.decode("latin-1").lower() not in ("content-length", "content-type"):
                out.raw_headers.append((key, value))
        return out


def mutate_app(app: FastAPI, mutations: tuple[str, ...]) -> FastAPI:
    """Wrap an existing shop so its HTML is rewritten by `mutations` on the way out."""
    unknown = set(mutations) - set(MUTATIONS)
    if unknown:
        raise ValueError(f"unknown mutation(s): {sorted(unknown)}")
    if mutations:
        app.add_middleware(_MutationMiddleware, mutations=mutations)
    return app


def create_mutated_app(
    db_path: str | Path,
    secret_key: str = "dev-only-secret",
    mutations: tuple[str, ...] = (),
) -> FastAPI:
    """A shop bound to one DB, serving its pages under `mutations` (empty = clean)."""
    return mutate_app(create_app(db_path, secret_key), mutations)
