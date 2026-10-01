"""
v3: infrastructure faults — break the plumbing on purpose, measure the recovery.

Mutations (app/mutations.py) perturb what the agent *sees*. Faults perturb what
the *infrastructure does*: a checkout that 500s, a response dropped after the
order already committed, a session that expires mid-task. Same mechanism — a
middleware wrapped around the untouched app — so the shop and its templates never
learn a fault is active.

The fault that matters most is `drop_checkout_response`: the server places the
order and empties the cart (the write commits), but the client gets a 500. An
agent that "recovers" by re-adding the items and checking out again places the
order a SECOND time. That double-order is invisible on the screen and obvious in
SQL — `exactly_one_new_order` is what catches it. v3's headline is how often
recovery double-orders.

Faults are a separate axis from mutations on purpose: a v3 cell can later carry
both (a mutated UI *and* a flaky checkout), but the two concerns stay decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

from app.main import create_app


@dataclass(frozen=True)
class Fault:
    """One named way the infrastructure misbehaves."""

    name: str
    description: str


FAULTS: dict[str, Fault] = {
    f.name: f
    for f in (
        Fault("error_on_checkout",
              "500 the checkout BEFORE it runs — no order is created"),
        Fault("drop_checkout_response",
              "let checkout commit the order, then 500 the client — the double-order trap"),
        Fault("expire_session",
              "strip the session so the agent is bounced to /login mid-task"),
    )
}

FAULT_NAMES: tuple[str, ...] = tuple(FAULTS)


def _is_checkout(request) -> bool:
    return request.method == "POST" and request.url.path == "/checkout"


class _FaultMiddleware(BaseHTTPMiddleware):
    """Inject faults around the real app. Outermost, so it can act before routing."""

    def __init__(self, app, faults: tuple[str, ...]) -> None:
        super().__init__(app)
        self._faults = frozenset(faults)

    async def dispatch(self, request, call_next):
        # Expire the session by dropping the cookie before SessionMiddleware reads
        # it: the agent looks logged out and gets redirected to /login.
        if "expire_session" in self._faults:
            request.scope["headers"] = [
                (k, v) for k, v in request.scope["headers"] if k.lower() != b"cookie"
            ]

        # 500 before the route runs: the order is never created.
        if "error_on_checkout" in self._faults and _is_checkout(request):
            return PlainTextResponse("Internal Server Error", status_code=500)

        response = await call_next(request)

        # The write already committed inside the route; we only lose the response.
        # The agent sees failure though the order exists — the double-order trap.
        if "drop_checkout_response" in self._faults and _is_checkout(request):
            return PlainTextResponse("Internal Server Error", status_code=500)

        return response


def inject_faults(app: FastAPI, faults: tuple[str, ...]) -> FastAPI:
    """Wrap an existing shop so `faults` fire on the matching requests."""
    unknown = set(faults) - set(FAULTS)
    if unknown:
        raise ValueError(f"unknown fault(s): {sorted(unknown)}")
    if faults:
        app.add_middleware(_FaultMiddleware, faults=faults)
    return app


def create_faulty_app(
    db_path: str | Path,
    secret_key: str = "dev-only-secret",
    faults: tuple[str, ...] = (),
) -> FastAPI:
    """A shop bound to one DB whose infrastructure misbehaves per `faults`."""
    return inject_faults(create_app(db_path, secret_key), faults)
