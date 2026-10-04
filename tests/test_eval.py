"""
C1 eval CLI + CI gate: the right exit code and a machine-readable summary, and the
unified `python -m bench` dispatcher forwards flags to the eval subcommand.
"""

from __future__ import annotations

import json

import pytest

import bench.__main__ as cli
from bench.eval import main as eval_main


def test_gate_passes_returns_zero(capsys):
    code = eval_main(["--agent", "oracle", "--tasks", "buy_blue_shirt",
                      "--conditions", "clean", "--trials", "1", "--min-pass-rate", "1.0"])
    assert code == 0
    assert "GATE: PASS" in capsys.readouterr().out


def test_gate_fails_returns_one(capsys):
    # Under faults the hardened agent loses error_on_checkout / expire_session, so a
    # 100% gate must fail the build.
    code = eval_main(["--agent", "hardened", "--tasks", "buy_blue_shirt",
                      "--conditions", "faults", "--trials", "1", "--min-pass-rate", "1.0"])
    assert code == 1
    assert "GATE: FAIL" in capsys.readouterr().out


def test_json_summary_is_machine_readable(capsys):
    code = eval_main(["--agent", "oracle", "--tasks", "buy_blue_shirt",
                      "--conditions", "clean", "--trials", "2", "--json", "--min-pass-rate", "1.0"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["gate_passed"] is True
    assert out["passed"] == out["planned"] == 2
    assert out["agent"] == "oracle" and out["cells"][0]["task"] == "buy_blue_shirt"


def test_report_is_written(tmp_path, capsys):
    path = tmp_path / "r.html"
    eval_main(["--agent", "oracle", "--tasks", "buy_blue_shirt",
               "--conditions", "clean", "--trials", "1", "--report", str(path)])
    assert path.exists() and path.read_text().startswith("<!doctype html>")


def test_unknown_task_errors():
    with pytest.raises(SystemExit):
        eval_main(["--tasks", "no_such_task", "--conditions", "clean", "--trials", "1"])


# ---------------------------------------------------------------- dispatcher

def test_dispatcher_forwards_flags_to_eval():
    code = cli.main(["eval", "--agent", "oracle", "--tasks", "buy_blue_shirt",
                     "--conditions", "clean", "--trials", "1", "--min-pass-rate", "1.0"])
    assert code == 0


def test_dispatcher_lists_commands_with_no_args(capsys):
    assert cli.main([]) == 2
    assert "commands:" in capsys.readouterr().out


def test_dispatcher_rejects_unknown_command(capsys):
    assert cli.main(["frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().out


def test_dispatcher_rejects_stray_flags_on_noarg_command(capsys):
    # selfcheck takes no options; a stray flag is an error, not silently ignored.
    assert cli.main(["selfcheck", "--bogus"]) == 2
    assert "takes no options" in capsys.readouterr().out
