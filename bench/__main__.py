"""
One CLI surface for the whole bench: `python -m bench <command> [args...]`.

Every tool in `bench/` exposes a `main()`; historically each ran as its own
`python -m bench.<module>`. This dispatcher unifies them behind a single front door
and forwards any remaining arguments to the chosen command (so `eval` gets its flags),
while each module's own `python -m bench.<module>` still works unchanged.

    python -m bench                                  # list commands
    python -m bench selfcheck                        # oracle-vs-noop calibration
    python -m bench eval --agent oracle --trials 1   # run + CI gate (args forwarded)
"""

from __future__ import annotations

import importlib
import inspect
import sys

# command -> (module exposing main(), one-line help). Order is the help order.
_COMMANDS: dict[str, tuple[str, str]] = {
    "eval": ("bench.eval", "run an agent across the matrix and gate on the result (CI surface)"),
    "selfcheck": ("bench.selfcheck", "oracle-vs-noop calibration on every task"),
    "run": ("bench.run", "run the oracle across the task × condition × trial matrix"),
    "gradecompare": ("bench.gradecompare", "score cheap graders (DOM/receipt/screenshot) vs SQL truth"),
    "hardened": ("bench.hardened", "naive vs hardened recovery under faults"),
    "report": ("bench.report", "write the static HTML robustness report to runs/"),
    "customer": ("bench.customer", "calibrate the customer-config layer worked example"),
}


def _print_help() -> None:
    print("usage: python -m bench <command> [args...]\n")
    print("Assay bench — own-the-website robustness tooling.\n")
    print("commands:")
    width = max(len(name) for name in _COMMANDS)
    for name, (_module, help_text) in _COMMANDS.items():
        print(f"  {name:<{width}}  {help_text}")
    print("\nRun 'python -m bench <command> --help' for a command's own options.")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return 0 if argv else 2

    command, rest = argv[0], argv[1:]
    if command not in _COMMANDS:
        print(f"unknown command: {command!r}\n")
        _print_help()
        return 2

    module = importlib.import_module(_COMMANDS[command][0])
    # Forward args only to commands whose main() accepts them (e.g. eval); the
    # no-arg tools reject stray flags rather than silently ignoring them.
    if inspect.signature(module.main).parameters:
        return module.main(rest)
    if rest:
        print(f"'{command}' takes no options, got: {' '.join(rest)}")
        return 2
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
