"""
The async DAG scheduler — the beating heart of a v1 run.

A run is a directed acyclic graph of `Node`s. `setup` builds the world once;
every case fans out from it; an `aggregate` node waits for all of them. This
module expands nothing and knows nothing about shops or graders — it just runs
a graph of async functions correctly, with the guarantees a robustness bench
actually needs:

    concurrency   a global cap, plus per-group caps (Solari's plan limit is one)
    retries       an infra hiccup gets another attempt with backoff
    timeouts      a per-attempt deadline, so one hung node can't stall the run
    isolation     one node failing never crashes the run; its dependents skip
    tolerance     a node can opt to run even when some dependencies failed

The one rule that matters most is not enforced here but *relied on* here:
a node function raises **only** on infrastructure failure and *returns
normally* when the real work finished — right or wrong. The engine retries
exceptions. So a browser that failed to launch (raise) is retried; an agent
that finished and bought the wrong shirt (return) is not. Retrying a wrong
answer would hide the flakiness the bench exists to measure.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

# A node function receives the results of its dependencies (keyed by node id)
# and returns anything. It may be sync or async; both are awaited correctly.
NodeFn = Callable[[dict[str, "NodeResult"]], Any]


@dataclass(frozen=True)
class Node:
    id: str
    fn: NodeFn
    deps: tuple[str, ...] = ()
    group: Optional[str] = None          # shares a per-group concurrency cap
    retries: int = 0                     # extra attempts after the first
    timeout_s: Optional[float] = None    # per-attempt deadline; None = no limit
    tolerate_dep_failure: bool = False    # run even if some deps failed/skipped


@dataclass
class NodeResult:
    id: str
    status: str                          # "success" | "failed" | "skipped"
    value: Any = None
    error: Optional[BaseException] = None
    attempts: int = 0                    # how many times fn was actually invoked

    @property
    def ok(self) -> bool:
        return self.status == "success"


@dataclass
class RunReport:
    """Every node's result, plus the sugar a caller usually wants."""

    results: dict[str, NodeResult] = field(default_factory=dict)

    def __getitem__(self, node_id: str) -> NodeResult:
        return self.results[node_id]

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.results

    @property
    def succeeded(self) -> list[str]:
        return [r.id for r in self.results.values() if r.status == "success"]

    @property
    def failed(self) -> list[str]:
        return [r.id for r in self.results.values() if r.status == "failed"]

    @property
    def skipped(self) -> list[str]:
        return [r.id for r in self.results.values() if r.status == "skipped"]


# ----------------------------------------------------------------- validation


def _validate(nodes: list[Node]) -> dict[str, Node]:
    index: dict[str, Node] = {}
    for n in nodes:
        if n.id in index:
            raise ValueError(f"duplicate node id: {n.id!r}")
        index[n.id] = n
    for n in nodes:
        for d in n.deps:
            if d not in index:
                raise ValueError(f"node {n.id!r} depends on unknown node {d!r}")
    _check_acyclic(index)
    return index


def _check_acyclic(index: dict[str, Node]) -> None:
    """DFS with a three-colour marking; raise on the first back-edge."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {nid: WHITE for nid in index}

    def visit(nid: str, stack: list[str]) -> None:
        colour[nid] = GREY
        for d in index[nid].deps:
            if colour[d] == GREY:
                cycle = stack[stack.index(d):] + [d]
                raise ValueError(f"dependency cycle: {' -> '.join(cycle)}")
            if colour[d] == WHITE:
                visit(d, stack + [d])
        colour[nid] = BLACK

    for nid in index:
        if colour[nid] == WHITE:
            visit(nid, [nid])


# --------------------------------------------------------------------- runner


async def run_dag(
    nodes: list[Node],
    *,
    max_concurrency: int = 8,
    group_limits: Optional[dict[str, int]] = None,
    backoff_base_s: float = 0.5,
    backoff_factor: float = 2.0,
) -> RunReport:
    """
    Run every node once its dependencies are terminal, honouring the caps.

    - A node runs when all its deps have *succeeded*.
    - If any dep failed or was skipped and the node is not
      `tolerate_dep_failure`, the node is **skipped** (the failure cascades).
    - A `tolerate_dep_failure` node runs regardless; it receives whatever
      results exist so it can decide what to do with the wreckage.

    Returns a RunReport with a NodeResult for *every* node — nothing is dropped.
    """
    group_limits = group_limits or {}
    index = _validate(nodes)

    results: dict[str, NodeResult] = {}
    global_sem = asyncio.Semaphore(max_concurrency)
    group_sems = {g: asyncio.Semaphore(limit) for g, limit in group_limits.items()}

    async def call(n: Node) -> Any:
        r = n.fn({d: results[d] for d in n.deps})
        if inspect.isawaitable(r):
            r = await r
        return r

    async def execute(n: Node) -> NodeResult:
        """Run one node with its semaphores, timeout, retries and backoff."""
        last_err: Optional[BaseException] = None
        for attempt in range(n.retries + 1):
            try:
                async with global_sem:
                    gsem = group_sems.get(n.group)
                    if gsem is not None:
                        async with gsem:
                            value = await _with_timeout(call(n), n.timeout_s)
                    else:
                        value = await _with_timeout(call(n), n.timeout_s)
                return NodeResult(n.id, "success", value=value, attempts=attempt + 1)
            except asyncio.CancelledError:
                raise
            except BaseException as e:  # infra failure — retry, then give up
                last_err = e
                if attempt < n.retries:
                    await asyncio.sleep(backoff_base_s * (backoff_factor ** attempt))
        return NodeResult(n.id, "failed", error=last_err, attempts=n.retries + 1)

    pending: set[str] = set(index)
    running: dict[str, asyncio.Task] = {}

    def deps_terminal(n: Node) -> bool:
        return all(d in results for d in n.deps)

    def deps_all_ok(n: Node) -> bool:
        return all(results[d].ok for d in n.deps)

    while pending or running:
        # Launch/skip everything currently ready. Skips resolve synchronously,
        # and one skip can make another node ready, so loop until it settles
        # before we ever await — otherwise a cascade of skips could look like a
        # deadlock.
        changed = True
        while changed:
            changed = False
            for nid in list(pending):
                n = index[nid]
                if not deps_terminal(n):
                    continue
                pending.discard(nid)
                changed = True
                if n.tolerate_dep_failure or deps_all_ok(n):
                    running[nid] = asyncio.create_task(execute(n))
                else:
                    results[nid] = NodeResult(nid, "skipped")

        if running:
            done, _ = await asyncio.wait(
                running.values(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                res = task.result()  # execute() never raises; it returns a result
                results[res.id] = res
                del running[res.id]
        elif pending:  # should be impossible — validation rules out cycles
            raise RuntimeError(f"deadlock: unresolved nodes {sorted(pending)}")

    return RunReport(results=results)


async def _with_timeout(coro: Awaitable, timeout_s: Optional[float]) -> Any:
    if timeout_s is None:
        return await coro
    return await asyncio.wait_for(coro, timeout_s)
