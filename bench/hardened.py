"""
A2: harden the agent loop — verify-before-stop and safe recovery.

The v1 agent (bench/agent.py) is a bare observe→decide→act loop: whatever the
brain says, it does, and when the brain says "stop" the run ends. That loop has
two failure modes the bench already quantifies:

  * **Premature stop.** The brain believes it finished and stops, but the goal
    isn't actually met. Nothing re-checks.
  * **Blind recovery.** On an error the naive strategy just retries the whole
    task. Under `drop_checkout_response` the order already committed, so the
    retry places a SECOND one — the 25% double-order rate in v3.

`HardenedPolicy` wraps any base `Policy` with two disciplines that attack exactly
those modes, using nothing but the browser the agent already has:

  * **verify-before-stop** — when the brain says done, don't take its word. Walk
    the task's evidence pages, read the actual state, and only stop if the goal
    is met. If it isn't, repair and re-check.
  * **safe recovery** — never repeat a state-changing action blind. The repair is
    *derived from the observed state*, so an order that already exists is never
    placed again. Check `/orders` before re-submitting checkout.

An honest note on scope. This shop has no *transient* fault (every fault fires on
every request), so the right recovery here is never "retry and hope" — it's
"tell already-succeeded from failed, and don't double-order." And the agent
verifies from the *page*, which (per v2) cannot see a side effect like an extra
order against a baseline it never knew — the exact blind spot that is why the
external SQL grader exists. Hardening fixes what's fixable from the agent's seat;
the SQL eval remains the truth.

Keyless: the brain here is a `ScriptedPolicy`; swap in `LLMPolicy` and the same
disciplines wrap it unchanged. Only the measurement of the pass^k gain waits on a key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from bench.agent import (
    Action,
    Observation,
    Policy,
    ScriptedPolicy,
    Step,
)
from bench.tasks import ALICE


# ------------------------------------------------------------- browser step builders
# The canonical ways to drive this shop, as Browser actions. Reused for both the
# base brain's script and the state-derived repair, so there's one source of truth
# for "how you place an order here."


def _login_steps() -> list[Action]:
    return [
        Action.goto("/login"),
        Action.fill("#email", ALICE[0]),
        Action.fill("#password", ALICE[1]),
        Action.click("text=Log in"),
    ]


def _add_steps(sku: str, qty: Optional[int]) -> list[Action]:
    steps = [Action.goto(f"/product/{sku}")]
    if qty is not None:
        steps.append(Action.fill("#qty", str(qty)))
    steps.append(Action.click("text=Add to cart"))
    return steps


def _checkout_steps(address: str) -> list[Action]:
    return [
        Action.goto("/checkout"),
        Action.fill("#ship_address", address),
        Action.click("text=Place order"),
    ]


# --------------------------------------------------------------------- task models
# What a purchase task's success looks like from the pages, and how to close the
# gap safely. `done` and `repair` both read only rendered page text (a dict of
# path -> visible text), the same vantage a real agent has.


def _order_count(text: str) -> int:
    # The orders table renders one "Order #<id>" link per order; the column header
    # is just "Order" (no '#'), so counting "order #" counts rows, not the header.
    return text.lower().count("order #")


def _cart_empty(text: str) -> bool:
    return "your cart is empty" in text.lower()


@dataclass(frozen=True)
class TaskModel:
    """The agent's own success model for a task, judged from page text."""

    evidence: tuple[str, ...]                               # pages to read to judge done-ness
    done: Callable[[dict[str, str]], tuple[bool, str]]      # (met?, detail) from {path: text}
    repair: Callable[[dict[str, str]], list[Action]]        # state-derived fix (safe by construction)


def _purchase_model(add_items: Sequence[tuple[str, Optional[int]]], address: str) -> TaskModel:
    def done(seen: dict[str, str]) -> tuple[bool, str]:
        n = _order_count(seen.get("/orders", ""))
        # One order on file means the purchase went through — even if the response
        # that would have told us was dropped. That's the anti-double-order check.
        return (n >= 1, f"{n} order(s) on file")

    def repair(seen: dict[str, str]) -> list[Action]:
        if _order_count(seen.get("/orders", "")) >= 1:
            return []                                       # SAFE: an order exists; never place another
        steps: list[Action] = []
        if _cart_empty(seen.get("/cart", "")):              # cart lost too — rebuild it
            for sku, qty in add_items:
                steps += _add_steps(sku, qty)
        steps += _checkout_steps(address)                   # otherwise just finish the checkout
        return steps

    return TaskModel(evidence=("/orders", "/cart"), done=done, repair=repair)


# The three purchase tasks — the ones a checkout fault can bite. (cancel/empty are
# left to the base brain: a checkout fault can't touch them, and their side-effect
# blind spot is SQL's job, not the agent's — see the module docstring.)
HARDENED_MODELS: dict[str, TaskModel] = {
    "buy_blue_shirt": _purchase_model([("SHIRT-BLU", None)], "12 Elm St, Austin, TX 78701"),
    "buy_socks_x3": _purchase_model([("SOCK-WOOL", 3)], "400 Main St, Dallas, TX 75201"),
    "buy_tote_and_mug": _purchase_model(
        [("TOTE-CNV", None), ("MUG-ENML", None)], "9 Oak Ave, Houston, TX 77002"
    ),
}


# Base-brain scripts: the competent-but-undisciplined agent. It does the task once
# and stops; the hardening is what makes it hold up when the stop is premature or
# the checkout response is lost.
def base_purchase_script(task_id: str) -> list[Action]:
    specs = {
        "buy_blue_shirt": (_login_steps() + _add_steps("SHIRT-BLU", None)
                           + _checkout_steps("12 Elm St, Austin, TX 78701")),
        "buy_socks_x3": (_login_steps() + _add_steps("SOCK-WOOL", 3)
                         + _checkout_steps("400 Main St, Dallas, TX 75201")),
        "buy_tote_and_mug": (_login_steps()
                             + [Action.goto("/cart"), Action.click("text=Remove Black Cap")]
                             + _add_steps("TOTE-CNV", None) + _add_steps("MUG-ENML", None)
                             + _checkout_steps("9 Oak Ave, Houston, TX 77002")),
    }
    return specs[task_id]


# -------------------------------------------------------------------- the hardened policy


class HardenedPolicy:
    """
    Wrap a base `Policy` so the loop verifies before it stops and recovers safely.

    State machine over the per-turn `decide` contract (it holds state across calls,
    like `ScriptedPolicy`):

        act      — delegate to the base brain; pass its actions through until it
                   says "stop", then begin a verification walk instead of stopping.
        assess   — goto each evidence page in turn, collect the rendered text, then
                   judge `done`. If met -> stop. If not and a round remains ->
                   compute a state-derived repair and run it. Else -> stop honestly.
        repair   — play the repair actions, then re-assess (bounded by max_rounds).

    The repair is a pure function of the pages just observed, so it can never
    duplicate a state change that already landed — that is the whole safety property.
    """

    def __init__(self, base: Policy, model: TaskModel, *, max_rounds: int = 1) -> None:
        self._base = base
        self._model = model
        self._max_rounds = max_rounds
        self._phase = "act"
        self._seen: dict[str, str] = {}
        self._walk: list[str] = []
        self._awaiting: Optional[str] = None
        self._repair: list[Action] = []
        self._round = 0

    def decide(self, instruction: str, observation: Observation, history: Sequence[Step]) -> Action:
        # Record the page we navigated to on the previous turn for assessment.
        if self._awaiting is not None:
            self._seen[self._awaiting] = observation.text
            self._awaiting = None

        if self._phase == "act":
            action = self._base.decide(instruction, observation, history)
            if action.kind != "stop":
                return action
            return self._begin_walk()                        # verify before trusting "done"

        if self._phase == "assess":
            if self._walk:
                return self._visit(self._walk.pop(0))
            ok, detail = self._model.done(self._seen)
            if ok:
                return Action.stop(f"verified: {detail}")
            if self._round < self._max_rounds:
                self._round += 1
                self._repair = self._model.repair(self._seen)
                if not self._repair:
                    return Action.stop(f"cannot recover safely: {detail}")
                self._phase = "repair"
                return self._repair.pop(0)
            return Action.stop(f"unverified after {self._max_rounds} round(s): {detail}")

        # self._phase == "repair"
        if self._repair:
            return self._repair.pop(0)
        return self._begin_walk()                            # re-check after repairing

    def _begin_walk(self) -> Action:
        self._phase = "assess"
        self._seen = {}
        self._walk = list(self._model.evidence)
        return self._visit(self._walk.pop(0))

    def _visit(self, path: str) -> Action:
        self._awaiting = path
        return Action.goto(path)


def hardened_policy_for(task, base: Optional[Policy] = None, *, max_rounds: int = 1) -> Policy:
    """
    The hardened agent for a task: a `ScriptedPolicy` base brain wrapped in
    `HardenedPolicy` when the task has a model, else the bare base brain.

    `base` is injectable so the same hardening can wrap an `LLMPolicy` (or a
    deliberately deficient brain, for tests) instead of the default script.
    """
    model = HARDENED_MODELS.get(task.id)
    if base is None:
        base = ScriptedPolicy(base_purchase_script(task.id)) if model else ScriptedPolicy([])
    return HardenedPolicy(base, model, max_rounds=max_rounds) if model else base


# -------------------------------------------------------------------- run.py integration


def hardened_recovery_run_fn(*, max_rounds: int = 1, max_steps: int = 40):
    """
    A bench/run.py `RunFn` that drives the hardened agent against the app composed
    for the cell's condition (clean -> mutated -> faulty). The browser analogue of
    `naive_recovery_run_fn`: it NEVER raises — recovery is the agent's job, not an
    engine retry — so a finished-but-wrong run is graded, not retried. The contrast
    with the naive version is the whole A2 measurement: same faults, no double-order.
    """
    from bench.agent import TestClientBrowser, run
    from bench.run import _compose_app

    async def run_fn(case, condition, trial):
        browser = TestClientBrowser(_compose_app(case.db_path, condition))
        policy = hardened_policy_for(case.task, max_rounds=max_rounds)
        instruction = case.task.instruction("http://shop")
        return await run(instruction, browser, policy, max_steps=max_steps)

    return run_fn


hardened_recovery_run_fn.__test__ = False


# ----------------------------------------------------------------------- CLI
# The A2 measurement, keyless: naive vs hardened recovery across the purchase tasks
# under every fault. The headline is the double-order rate — naive's blind retry
# vs the hardened agent's state-checked one — and the drop cells the hardening wins.


def main() -> int:
    import asyncio
    import tempfile

    from bench.env import build_seed, prepare_case
    from bench.grader import grade
    from bench.run import fault_conditions, naive_recovery_run_fn, recovery_outcome
    from bench.tasks import BUY_BLUE_SHIRT, BUY_SOCKS_X3, BUY_TOTE_AND_MUG

    purchase = (BUY_BLUE_SHIRT, BUY_SOCKS_X3, BUY_TOTE_AND_MUG)
    conds = fault_conditions()
    strategies = {
        "naive": lambda case, cond: naive_recovery_run_fn()(case, cond, 0),
        "hardened": lambda case, cond: asyncio.run(hardened_recovery_run_fn()(case, cond, 0)),
    }

    with tempfile.TemporaryDirectory() as tmp:
        seed = build_seed(f"{tmp}/seed.db")
        tallies: dict[str, dict[str, int]] = {}
        print(f"{'strategy':<10} {'task':<18} {'condition':<28} {'orders':>7} {'pass':>5}")
        print("-" * 72)
        for name, drive in strategies.items():
            t = tallies.setdefault(name, {"double": 0, "pass": 0, "cells": 0})
            for task in purchase:
                for cond in conds:
                    case = prepare_case(task, f"{tmp}/{name}/{task.id}/{cond.name}", seed)
                    drive(case, cond)
                    outcome = recovery_outcome(case)
                    passed = grade(case).passed
                    t["cells"] += 1
                    t["double"] += outcome == "double"
                    t["pass"] += passed
                    print(f"{name:<10} {task.id:<18} {cond.name:<28} "
                          f"{outcome:>7} {('yes' if passed else 'no'):>5}")
            print("-" * 72)

    print()
    for name, t in tallies.items():
        print(f"{name:<10} double-order {t['double']}/{t['cells']} "
              f"({t['double']/t['cells']:.0%})   passed {t['pass']}/{t['cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
