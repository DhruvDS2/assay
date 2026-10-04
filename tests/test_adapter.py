"""
C1 adapter: an agent spec resolves to a RunFn — built-in, or a customer's own
`module:factory` entrypoint — and the result actually drives the shop + SQL grader.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from bench.adapter import BUILTINS, load_entrypoint, policy_run_fn, resolve_agent
from bench.agent import Action, ScriptedPolicy
from bench.env import prepare_case
from bench.grader import grade
from bench.hardened import _login_steps
from bench.run import CLEAN
from bench.tasks import BUY_BLUE_SHIRT, EMPTY_CART


def _drive(run_fn, task, seed, tmp_path):
    case = prepare_case(task, tmp_path / task.id, seed)
    result = run_fn(case, CLEAN, 0)      # RunFn may be sync (oracle) or async (agent loop)
    if inspect.isawaitable(result):
        asyncio.run(result)
    return grade(case)


def test_builtin_oracle_resolves_and_passes(seed, tmp_path):
    run_fn = resolve_agent("oracle")
    assert _drive(run_fn, BUY_BLUE_SHIRT, seed, tmp_path).passed


def test_builtin_hardened_resolves_and_passes(seed, tmp_path):
    run_fn = resolve_agent("hardened")
    assert _drive(run_fn, BUY_BLUE_SHIRT, seed, tmp_path).passed


def test_llm_resolves_without_a_key(seed, tmp_path):
    """Resolving 'llm' must not touch the network — the key is only needed at run."""
    assert callable(resolve_agent("llm"))
    assert "llm" in BUILTINS


def test_entrypoint_module_factory_resolves_and_runs(seed, tmp_path):
    # bench.run:_testclient_oracle_run_fn is a zero-arg factory that returns a RunFn —
    # exactly the BYO contract, so it exercises the 'module:factory' path for real.
    run_fn = resolve_agent("bench.run:_testclient_oracle_run_fn")
    assert _drive(run_fn, BUY_BLUE_SHIRT, seed, tmp_path).passed


def test_policy_run_fn_wraps_a_policy(seed, tmp_path):
    """A Policy-based agent becomes a RunFn via the exported helper."""
    def make_policy(case):
        return ScriptedPolicy(_login_steps() + [
            Action.goto("/cart"),
            Action.click("text=Empty cart"),
            Action.stop("cart emptied"),
        ])

    run_fn = policy_run_fn(make_policy)
    assert _drive(run_fn, EMPTY_CART, seed, tmp_path).passed


def test_load_entrypoint_errors_are_clear():
    with pytest.raises(ValueError, match="expected 'module:attr'"):
        load_entrypoint("no_colon_here")
    with pytest.raises(ValueError, match="cannot import module"):
        load_entrypoint("bench.does_not_exist:thing")
    with pytest.raises(ValueError, match="has no attribute"):
        load_entrypoint("bench.adapter:not_a_real_attr")


def test_unknown_agent_name_raises():
    with pytest.raises(ValueError, match="unknown agent"):
        resolve_agent("definitely-not-an-agent")


def test_entrypoint_factory_must_return_callable():
    # bench.adapter:BUILTINS is a tuple, not a callable factory -> clear error.
    with pytest.raises(ValueError, match="not callable"):
        resolve_agent("bench.adapter:BUILTINS")
