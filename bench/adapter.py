"""
C1: bring-your-own-agent adapter — the seam a customer plugs their agent into.

The bench runs an agent through a `RunFn`: an async `(case, condition, trial) ->
Transcript` that drives the agent and leaves the result in the case's database for
the SQL grader to read. This module resolves an "agent spec" string to a `RunFn`,
so the eval CLI (and a GitHub Action) can point the harness at ANY agent without a
code change:

  - a built-in name — "oracle" (scripted perfect run), "hardened" (the A2 recovery
    agent), or "llm" (the real LLMPolicy; needs ANTHROPIC_API_KEY at run time).
  - a Python entrypoint "module:factory" — your own code. `factory()` must return a
    fresh `RunFn`. If your agent is a `Policy`, wrap it in one line with the exported
    `policy_run_fn(make_policy)` and return that.

Keyless by default: "oracle" and "hardened" need no key; only "llm" does, and even
then only when a case actually runs — resolving the spec never touches the network.
"""

from __future__ import annotations

import importlib
from typing import Callable

# A RunFn is (case, condition, trial) -> awaitable[Transcript]. Kept as a bare alias
# so importing this module does not drag in bench.run / app at load time.
RunFn = Callable

BUILTINS: tuple[str, ...] = ("oracle", "hardened", "llm")


def policy_run_fn(make_policy: Callable, *, max_steps: int = 25) -> RunFn:
    """
    Turn a per-case `Policy` factory into a `RunFn` that drives it against the shop
    mutated for the cell's condition. The BYO helper for Policy-based agents:

        def make_agent(case):
            return MyPolicy(...)
        RUN = policy_run_fn(make_agent)      # expose RUN; point --agent at "mypkg:RUN"

    `make_policy(case)` is called once per case, so stateful policies get a fresh
    instance per run (never shared across cases).
    """
    from app.mutations import create_mutated_app
    from bench.agent import TestClientBrowser, run

    async def run_fn(case, condition, trial):
        app = create_mutated_app(case.db_path, mutations=condition.mutations)
        browser = TestClientBrowser(app)
        instruction = case.task.instruction("http://shop")
        return await run(instruction, browser, make_policy(case), max_steps=max_steps)

    return run_fn


def _builtin(name: str, *, model: str | None = None) -> RunFn | None:
    """Resolve a built-in agent name to its RunFn, or None if the name isn't one."""
    if name == "oracle":
        from bench.run import _testclient_oracle_run_fn

        return _testclient_oracle_run_fn()
    if name == "hardened":
        from bench.hardened import hardened_recovery_run_fn

        return hardened_recovery_run_fn()
    if name == "llm":
        from bench.agent import llm_run_fn

        return llm_run_fn(model=model) if model else llm_run_fn()
    return None


def load_entrypoint(spec: str):
    """Import `module:attr` and return `attr`, with a clear error on each failure."""
    if ":" not in spec:
        raise ValueError(f"not an entrypoint (expected 'module:attr'): {spec!r}")
    mod_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(mod_name)
    except ImportError as e:
        raise ValueError(f"cannot import module {mod_name!r} from agent spec {spec!r}: {e}") from e
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ValueError(f"module {mod_name!r} has no attribute {attr!r} (agent spec {spec!r})") from e


def resolve_agent(spec: str, *, model: str | None = None) -> RunFn:
    """
    Resolve an agent spec to a `RunFn`.

    `spec` is a built-in name (see `BUILTINS`) or a `module:factory` entrypoint whose
    `factory()` returns a fresh `RunFn`. Raises `ValueError` with an actionable
    message for an unknown name, an unimportable entrypoint, or a factory that does
    not return a callable.
    """
    built = _builtin(spec, model=model)
    if built is not None:
        return built
    if ":" in spec:
        factory = load_entrypoint(spec)
        if not callable(factory):
            raise ValueError(f"agent entrypoint {spec!r} is not callable")
        run_fn = factory()
        if not callable(run_fn):
            raise ValueError(f"agent factory {spec!r} must return a RunFn (a callable)")
        return run_fn
    raise ValueError(
        f"unknown agent {spec!r}; use one of {BUILTINS} or a 'module:factory' entrypoint"
    )
