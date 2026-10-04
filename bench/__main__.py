"""
One CLI surface for the whole bench: `python -m bench <command>`.

Every tool in `bench/` already exposes a `main() -> int`; historically each was
run as its own `python -m bench.<module>`. This dispatcher unifies them behind a
single argparse entrypoint so there is one discoverable command list, while each
module's own `python -m bench.<module>` still works unchanged (this only adds a
front door, it doesn't move the rooms).

    python -m bench                 # list commands
    python -m bench selfcheck       # oracle-vs-noop calibration
    python -m bench gradecompare    # cheap graders scored against SQL truth
"""

from __future__ import annotations

import argparse
import importlib

# command -> (module exposing main(), one-line help). Order is the help order.
_COMMANDS: dict[str, tuple[str, str]] = {
    "selfcheck": ("bench.selfcheck", "oracle-vs-noop calibration on every task"),
    "run": ("bench.run", "run the oracle across the task × condition × trial matrix"),
    "gradecompare": ("bench.gradecompare", "score cheap graders (DOM/receipt/screenshot) vs SQL truth"),
    "hardened": ("bench.hardened", "naive vs hardened recovery under faults"),
    "report": ("bench.report", "write the static HTML robustness report to runs/"),
    "customer": ("bench.customer", "calibrate the customer-config layer worked example"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        description="Assay bench — own-the-website robustness tooling.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name, (_module, help_text) in _COMMANDS.items():
        sub.add_parser(name, help=help_text)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2

    module = importlib.import_module(_COMMANDS[args.command][0])
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
