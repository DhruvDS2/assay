"""
The scheduler's own tests, driven by fake sleep-nodes.

The README's rule: test the engine with fake nodes before anything real
touches it. These nodes just sleep, count, or raise — no shop, no grader — so
a failure here is a bug in the engine, never in the bench.
"""

from __future__ import annotations

import asyncio

import pytest

from bench.engine import Node, RunReport, run_dag


def run(nodes, **kw) -> RunReport:
    return asyncio.run(run_dag(nodes, **kw))


# ---------------------------------------------------------------- fake nodes


def const(value):
    """A node that does nothing but return a value."""
    async def fn(deps):
        return value
    return fn


def boom(exc=RuntimeError("infra failed")):
    """A node that always raises — stands in for an infra failure."""
    async def fn(deps):
        raise exc
    return fn


def flaky(fail_times, value="ok"):
    """Raises the first `fail_times` invocations, then succeeds. Mutable count."""
    state = {"n": 0}
    async def fn(deps):
        if state["n"] < fail_times:
            state["n"] += 1
            raise RuntimeError(f"transient #{state['n']}")
        return value
    return fn


# ---------------------------------------------------------------- basic shape


def test_linear_chain_runs_and_passes_results():
    def add_one(deps):
        return deps["a"].value + 1
    nodes = [
        Node("a", const(1)),
        Node("b", add_one, deps=("a",)),
        Node("c", lambda deps: deps["b"].value * 10, deps=("b",)),
    ]
    r = run(nodes)
    assert r["a"].value == 1
    assert r["b"].value == 2
    assert r["c"].value == 20
    assert r.succeeded == ["a", "b", "c"] or set(r.succeeded) == {"a", "b", "c"}


def test_sync_and_async_fns_both_work():
    nodes = [
        Node("sync", lambda deps: 41),
        Node("async", const(1)),
        Node("sum", lambda deps: deps["sync"].value + deps["async"].value,
             deps=("sync", "async")),
    ]
    r = run(nodes)
    assert r["sum"].value == 42


def test_every_node_gets_a_result():
    nodes = [Node("a", const(1)), Node("b", boom()), Node("c", const(3))]
    r = run(nodes)
    assert set(r.results) == {"a", "b", "c"}


# ---------------------------------------------------------------- concurrency


def test_global_concurrency_is_capped():
    live = {"now": 0, "max": 0}

    def tracker():
        async def fn(deps):
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
            await asyncio.sleep(0.05)
            live["now"] -= 1
            return True
        return fn

    nodes = [Node(f"n{i}", tracker()) for i in range(10)]
    run(nodes, max_concurrency=3)
    assert live["max"] == 3


def test_per_group_cap_is_independent_of_global():
    live = {"now": 0, "max": 0}

    def tracker():
        async def fn(deps):
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
            await asyncio.sleep(0.05)
            live["now"] -= 1
        return fn

    # 6 solari nodes, global cap 10, group cap 2 -> never more than 2 at once.
    nodes = [Node(f"s{i}", tracker(), group="solari") for i in range(6)]
    run(nodes, max_concurrency=10, group_limits={"solari": 2})
    assert live["max"] == 2


def test_independent_nodes_actually_run_in_parallel():
    # 4 nodes, 0.1s each, cap 4 -> wall clock well under the 0.4s serial sum.
    nodes = [Node(f"n{i}", _sleep(0.1)) for i in range(4)]

    async def timed():
        loop = asyncio.get_event_loop()
        start = loop.time()
        await run_dag(nodes, max_concurrency=4)
        return loop.time() - start

    assert asyncio.run(timed()) < 0.3


def _sleep(seconds, value=None):
    async def fn(deps):
        await asyncio.sleep(seconds)
        return value
    return fn


# ---------------------------------------------------------------- retries


def test_flaky_node_is_retried_until_it_succeeds():
    node = Node("f", flaky(fail_times=2), retries=3)
    r = run([node], backoff_base_s=0)
    assert r["f"].status == "success"
    assert r["f"].attempts == 3  # two failures + one success


def test_retries_are_exhausted_then_the_node_fails():
    node = Node("f", flaky(fail_times=5), retries=2)
    r = run([node], backoff_base_s=0)
    assert r["f"].status == "failed"
    assert r["f"].attempts == 3  # first try + two retries
    assert isinstance(r["f"].error, RuntimeError)


def test_a_returning_node_is_never_retried():
    # The core rule: an agent that *finishes* (returns) is data, not an error.
    # Only raising nodes are retried, so a normal return is a single attempt.
    calls = {"n": 0}

    def counted():
        async def fn(deps):
            calls["n"] += 1
            return "agent bought the wrong shirt"  # wrong, but a clean finish
        return fn

    r = run([Node("agent", counted(), retries=5)], backoff_base_s=0)
    assert r["agent"].status == "success"
    assert r["agent"].attempts == 1
    assert calls["n"] == 1


# ---------------------------------------------------------------- timeouts


def test_a_hung_node_times_out_and_is_retried():
    # First attempt hangs past the deadline; the retry is fast and succeeds.
    state = {"n": 0}

    async def fn(deps):
        state["n"] += 1
        if state["n"] == 1:
            await asyncio.sleep(10)  # will be cancelled by the timeout
        return "recovered"

    node = Node("slow", lambda deps: fn(deps), timeout_s=0.05, retries=1)
    r = run([node], backoff_base_s=0)
    assert r["slow"].status == "success"
    assert r["slow"].attempts == 2


# ---------------------------------------------------------------- isolation


def test_failure_is_isolated_and_cascades_to_dependents():
    nodes = [
        Node("ok", const(1)),
        Node("bad", boom()),
        Node("needs_bad", const(2), deps=("bad",)),
        Node("needs_ok", const(3), deps=("ok",)),
    ]
    r = run(nodes)
    assert r["ok"].status == "success"
    assert r["bad"].status == "failed"
    assert r["needs_bad"].status == "skipped"   # cascaded
    assert r["needs_ok"].status == "success"    # unaffected sibling still ran


def test_skip_cascades_through_multiple_levels():
    nodes = [
        Node("root", boom()),
        Node("mid", const(1), deps=("root",)),
        Node("leaf", const(2), deps=("mid",)),
    ]
    r = run(nodes)
    assert r["root"].status == "failed"
    assert r["mid"].status == "skipped"
    assert r["leaf"].status == "skipped"


def test_aggregate_tolerates_dep_failure_and_sees_the_wreckage():
    # Mirrors the real aggregate node: it must run even when cases failed,
    # and it must be able to inspect which ones did.
    def summarise(deps):
        return {k: v.status for k, v in deps.items()}

    nodes = [
        Node("setup", const("snapshot")),
        Node("case1", const("pass"), deps=("setup",)),
        Node("case2", boom(), deps=("setup",)),
        Node("aggregate", summarise, deps=("case1", "case2"),
             tolerate_dep_failure=True),
    ]
    r = run(nodes)
    assert r["aggregate"].status == "success"
    assert r["aggregate"].value == {"case1": "success", "case2": "failed"}


# ---------------------------------------------------------------- validation


def test_cycle_is_rejected():
    nodes = [
        Node("a", const(1), deps=("c",)),
        Node("b", const(2), deps=("a",)),
        Node("c", const(3), deps=("b",)),
    ]
    with pytest.raises(ValueError, match="cycle"):
        run(nodes)


def test_unknown_dependency_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        run([Node("a", const(1), deps=("ghost",))])


def test_duplicate_id_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        run([Node("a", const(1)), Node("a", const(2))])
