"""
C1: the eval CLI + CI gate — `python -m bench eval --agent <spec> [gate flags]`.

Runs an agent across a condition set through the real engine, prints the honest
matrix summary (the same `RunSummary` the research harness uses), optionally writes
the HTML report, and returns a CI-friendly exit code: 0 when the run clears the gate,
1 when it doesn't. This is the surface a GitHub Action calls to fail a build when an
agent regresses — the eval certifying the agent, in one command.

The gate is deliberately explicit, not magic: `--min-pass-rate` thresholds
passed/planned, and infra-skips fail the build unless `--allow-skips` is set (a
skipped cell means the agent never got a fair shot — silent in a bare pass count,
loud here).

    python -m bench eval --agent oracle --conditions seven --trials 1 --min-pass-rate 1.0
    python -m bench eval --agent mypkg:make_agent --report runs/report.html --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile


def _conditions(name: str):
    from bench.run import CLEAN, fault_conditions, seven_conditions

    return {"clean": (CLEAN,), "seven": seven_conditions(), "faults": fault_conditions()}[name]


def _select_tasks(csv: str):
    from bench.tasks import TASKS

    if not csv:
        return TASKS
    want = {s.strip() for s in csv.split(",") if s.strip()}
    by_id = {t.id: t for t in TASKS}
    unknown = want - set(by_id)
    if unknown:
        raise SystemExit(f"unknown task id(s): {sorted(unknown)}; available: {sorted(by_id)}")
    return tuple(t for t in TASKS if t.id in want)   # preserve TASKS order


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bench eval",
        description="Run an agent across the matrix and gate on the SQL-truth result.",
    )
    p.add_argument("--agent", default="oracle",
                   help="built-in (oracle|hardened|llm) or a 'module:factory' entrypoint")
    p.add_argument("--model", default=None, help="model id for the llm agent")
    p.add_argument("--conditions", choices=["clean", "seven", "faults"], default="seven")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    p.add_argument("--report", default=None, help="write the HTML report to this path")
    p.add_argument("--min-pass-rate", type=float, default=None,
                   help="fail (exit 1) if passed/planned is below this [0..1]")
    p.add_argument("--allow-skips", action="store_true",
                   help="do not fail the build on infra-skipped cells")
    p.add_argument("--json", action="store_true", help="emit a machine-readable summary")
    return p


def _gate(summary, *, min_pass_rate, allow_skips) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ppp = summary.passed_over_planned
    if min_pass_rate is not None and (ppp is None or ppp < min_pass_rate):
        reasons.append(f"pass-rate {ppp if ppp is None else f'{ppp:.0%}'} < required {min_pass_rate:.0%}")
    if summary.skipped and not allow_skips:
        reasons.append(f"{summary.skipped} infra-skipped cell(s) (use --allow-skips to tolerate)")
    return (not reasons), reasons


def _summary_dict(summary, args, gate_ok, reasons) -> dict:
    return {
        "agent": args.agent,
        "conditions": args.conditions,
        "trials": args.trials,
        "planned": summary.planned,
        "attempted": summary.attempted,
        "passed": summary.passed,
        "skipped": summary.skipped,
        "passed_over_planned": summary.passed_over_planned,
        "passed_over_attempted": summary.passed_over_attempted,
        "reliable_cells": len(summary.reliable_cells),
        "total_cells": len(summary.cells),
        "gate_passed": gate_ok,
        "gate_failures": reasons,
        "cells": [
            {"task": c.task_id, "condition": c.condition, "passed": c.passed,
             "attempted": c.attempted, "planned": c.planned, "pass_hat_k": c.pass_hat_k}
            for c in summary.cells
        ],
    }


def main(argv: list[str] | None = None) -> int:
    from bench.adapter import resolve_agent
    from bench.run import run_matrix

    args = _build_parser().parse_args(argv)
    run_fn = resolve_agent(args.agent, model=args.model)
    tasks = _select_tasks(args.tasks)
    conditions = _conditions(args.conditions)

    with tempfile.TemporaryDirectory() as tmp:
        summary, _report = asyncio.run(
            run_matrix(tasks, conditions, args.trials, root=tmp, run_fn=run_fn)
        )

    gate_ok, reasons = _gate(summary, min_pass_rate=args.min_pass_rate, allow_skips=args.allow_skips)

    if args.report:
        from bench.report import write_report

        write_report(summary, args.report,
                     title="Assay — agent eval",
                     subtitle=f"agent={args.agent} · conditions={args.conditions} · trials={args.trials}")

    if args.json:
        print(json.dumps(_summary_dict(summary, args, gate_ok, reasons), indent=2))
    else:
        print(summary.format())
        print()
        print(f"agent={args.agent}  conditions={args.conditions}  trials={args.trials}")
        if args.report:
            print(f"report: {args.report}")
        print("GATE: PASS" if gate_ok else "GATE: FAIL — " + "; ".join(reasons))

    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
