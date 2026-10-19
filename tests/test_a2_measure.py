"""
A2, MEASURED — keyless, deterministic proof of the recovery win on the FAULT axis.

Two things this module locks down, with no `anthropic` call and no API key:

  1. **The hardening wraps the real brain, constructed keyless.** `HardenedLLMPolicy`
     is a `HardenedPolicy` whose base is an `LLMPolicy`. We build it with an injected
     fake client (and also prove the wrapper's disciplines still fire by driving a
     deliberately-deficient fake model through the real shop). No network, ever.

  2. **naive double-orders under `drop_checkout_response`; hardened lifts that cell
     to zero.** We assert it three ways: directly against the shop + fault, through
     the A/B `measure_recovery` tally (per-cell double-order rate), and through the
     engine via `run_matrix` — the drop cell fails for naive and passes for hardened,
     with nothing infra-skipped. This is A2's win, on the axis A1's 73/105 mutation
     matrix never touched.

Determinism: every agent here is scripted (the real v3 run_fns drive a
`ScriptedPolicy`; the one LLM path uses a `FakeClient` with a fixed reply list).
Nothing depends on a live model's choices.

Run: python -m pytest tests/test_a2_measure.py -q
"""

from __future__ import annotations

import asyncio

from bench.a2_measure import (
    CellResult,
    HardenedLLMPolicy,
    Measurement,
    PURCHASE_TASKS,
    StrategyTally,
    hardened_llm_policy_for,
    measure_recovery,
)
from bench.agent import Action, Observation, TestClientBrowser, run
from bench.env import prepare_case
from bench.grader import grade
from bench.hardened import HARDENED_MODELS
from bench.run import (
    CLEAN,
    Condition,
    _compose_app,
    fault_conditions,
    naive_recovery_run_fn,
    recovery_outcome,
    run_matrix,
)
from bench.tasks import BUY_BLUE_SHIRT, CANCEL_LATEST_ORDER

DROP = Condition("drop", faults=("drop_checkout_response",))
DROP_COND_NAME = "fault_drop_checkout_response"


# ----------------------------------------------------------- a fake Claude client
# Mirrors tests/test_llm_agent.py so an LLMPolicy runs with zero network. A turn
# with no tool call makes the policy "stop" — the premature-stop a deficient model
# produces, which the hardening is meant to catch.


class _Block:
    def __init__(self, type_, name=None, input=None, text=""):
        self.type, self.name, self.input, self.text = type_, name, input, text


class _Message:
    def __init__(self, blocks):
        self.content = blocks


class _Messages:
    def __init__(self, replies):
        self._replies = list(replies)
        self._i = 0
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        return reply


class FakeClient:
    def __init__(self, replies):
        self.messages = _Messages(replies)


def _stop_reply():
    """A model that returns no tool call -> LLMPolicy turns it into stop()."""
    return _Message([_Block("text", text="I'm done.")])


# ============================================================ (1) keyless construction


def test_hardened_llm_policy_is_a_hardened_policy_around_an_llm_brain():
    """Construction only: the wrapper is a HardenedPolicy; its base is an LLMPolicy."""
    from bench.agent import LLMPolicy
    from bench.hardened import HardenedPolicy

    policy = HardenedLLMPolicy(HARDENED_MODELS["buy_blue_shirt"], client=FakeClient([]))
    assert isinstance(policy, HardenedPolicy)
    assert isinstance(policy._base, LLMPolicy)      # the real brain, wrapped unchanged


def test_hardened_llm_policy_constructs_without_network_or_key():
    """Building the policy makes no call: a FakeClient recording calls stays untouched."""
    fake = FakeClient([])
    HardenedLLMPolicy(HARDENED_MODELS["buy_socks_x3"], client=fake)
    assert fake.messages.calls == []                # construction is inert


def test_factory_wraps_purchase_tasks_and_bare_brains_the_rest():
    purchase = hardened_llm_policy_for(BUY_BLUE_SHIRT, client=FakeClient([]))
    assert isinstance(purchase, HardenedLLMPolicy)

    from bench.agent import LLMPolicy

    # A checkout fault can't touch cancel/empty, so there's nothing to harden.
    non_purchase = hardened_llm_policy_for(CANCEL_LATEST_ORDER, client=FakeClient([]))
    assert isinstance(non_purchase, LLMPolicy)
    assert not isinstance(non_purchase, HardenedLLMPolicy)


def test_verify_before_stop_fires_around_the_real_brain_offline(seed, tmp_path):
    """
    A fake model that logs in and immediately 'stops' (returns no tool call) would
    leave the task unfinished. Wrapped in HardenedLLMPolicy, the loop re-reads
    /orders, sees none, completes the checkout — all keyless, all deterministic.
    """
    case = prepare_case(BUY_BLUE_SHIRT, tmp_path / "verify", seed)
    client = FakeClient([
        _act("goto", "/login"),
        _act("fill", "#email", "alice@example.com"),
        _act("fill", "#password", "password123"),
        _act("click", "text=Log in"),
        _stop_reply(),                     # premature: brain gives up before ordering
    ])
    policy = HardenedLLMPolicy(HARDENED_MODELS["buy_blue_shirt"], client=client)
    browser = TestClientBrowser(_compose_app(case.db_path, CLEAN))
    asyncio.run(run(BUY_BLUE_SHIRT.instruction("http://shop"), browser, policy, max_steps=40))
    assert recovery_outcome(case) == "ok"           # hardening finished the job
    assert grade(case).passed


def _act(kind, target="", value="", note=""):
    return _Message([_Block("tool_use", name="act",
                            input={"kind": kind, "target": target,
                                   "value": value, "note": note})])


# ================================================= (2) naive double-orders, hardened does not


def test_naive_double_orders_under_drop_hardened_does_not(seed, tmp_path):
    """Direct A/B against the real shop + the drop fault, keyless (scripted brains)."""
    naive = prepare_case(BUY_BLUE_SHIRT, tmp_path / "naive", seed)
    naive_recovery_run_fn()(naive, DROP, 0)
    assert recovery_outcome(naive) == "double"      # blind retry places a 2nd order
    assert not grade(naive).passed                  # exactly_one_new_order catches it


def test_measure_recovery_tally_shows_the_double_order_gap(seed, tmp_path):
    """
    The A/B tally over the FAULT axis: naive double-orders on the drop cells,
    hardened never does. (The axis A1's mutation-only 73/105 never exercised.)
    """
    m = measure_recovery(root=tmp_path, seed_path=seed)
    assert isinstance(m, Measurement)

    # Per CLAUDE.md the drop fault is the double-order trap; naive hits it on every
    # purchase task, hardened on none.
    assert m.naive.double_orders_on(DROP_COND_NAME) == len(PURCHASE_TASKS)
    assert m.hardened.double_orders_on(DROP_COND_NAME) == 0

    # Totals: hardened strictly fewer double-orders than naive, and zero across all.
    assert m.hardened.double_orders == 0
    assert m.naive.double_orders >= len(PURCHASE_TASKS)
    assert m.hardened.double_orders < m.naive.double_orders

    # Every drop cell flips pass/fail accordingly.
    for task in PURCHASE_TASKS:
        assert not m.cell("naive", task.id, DROP_COND_NAME).passed
        assert m.cell("hardened", task.id, DROP_COND_NAME).passed

    # Baseline sanity: both strategies succeed on the clean fault-axis baseline.
    for task in PURCHASE_TASKS:
        assert m.cell("naive", task.id, "clean").outcome == "ok"
        assert m.cell("hardened", task.id, "clean").outcome == "ok"


def test_tally_double_rate_is_well_defined():
    """The headline rate is double_orders / cells, None only when empty."""
    cells = (
        CellResult("naive", "t", "clean", "ok", True),
        CellResult("naive", "t", DROP_COND_NAME, "double", False),
    )
    tally = StrategyTally("naive", cells)
    assert tally.double_orders == 1
    assert tally.double_rate == 0.5
    assert StrategyTally("naive", ()).double_rate is None


# ======================================================= (3) through the engine (run_matrix)


def test_naive_fails_drop_cell_through_the_engine(seed, tmp_path):
    """run_matrix with the naive run_fn: the drop cell fails (the double-order)."""
    summary, _report = asyncio.run(
        run_matrix(
            PURCHASE_TASKS,
            fault_conditions(),
            trials=1,
            root=tmp_path / "naive_matrix",
            run_fn=lambda case, cond, trial: naive_recovery_run_fn()(case, cond, trial),
        )
    )
    by = {(c.task_id, c.condition): c for c in summary.cells}
    for task in PURCHASE_TASKS:
        assert by[(task.id, DROP_COND_NAME)].passed == 0          # naive double-orders -> fail
    assert summary.skipped == 0                                   # a wrong agent, never skipped


def test_hardened_lifts_the_drop_cell_to_zero_through_the_engine(seed, tmp_path):
    """
    The same matrix with the hardened run_fn: every drop cell passes (one order,
    not two), clean still works, nothing infra-skipped. This is the A2 win, read
    through the real engine — not the worker's self-report.
    """
    from bench.hardened import hardened_recovery_run_fn

    summary, _report = asyncio.run(
        run_matrix(
            PURCHASE_TASKS,
            fault_conditions(),
            trials=1,
            root=tmp_path / "hardened_matrix",
            run_fn=hardened_recovery_run_fn(),
        )
    )
    by = {(c.task_id, c.condition): c for c in summary.cells}
    for task in PURCHASE_TASKS:
        assert by[(task.id, "clean")].passed == 1                 # baseline holds
        assert by[(task.id, DROP_COND_NAME)].passed == 1          # recovered, not doubled
    assert summary.skipped == 0


def test_hardened_llm_run_fn_resolves_through_the_cli_keyless():
    """The eval CLI entrypoint for the hardened real-brain agent builds without a key
    (the model client is acquired lazily at decide time, not here)."""
    import inspect

    from bench.adapter import resolve_agent
    from bench.a2_measure import hardened_llm_run_fn

    run_fn = resolve_agent("bench.a2_measure:hardened_llm_run_fn")
    assert callable(run_fn)
    # the agent-loop RunFn is async (the engine awaits it), unlike the sync oracle
    assert inspect.iscoroutinefunction(run_fn)
    assert callable(hardened_llm_run_fn())
