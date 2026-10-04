# CLI & CI (C1)

The eval/CI surface: one command to run an agent through the bench, a bring-your-own-agent
seam, and a GitHub Action that fails a build when an agent regresses. All of it reads the
same SQL ground truth the research harness does — the eval certifies the agent, the agent
never certifies itself.

## The unified CLI

Every tool lives behind one front door:

```bash
python -m bench                 # list commands
python -m bench selfcheck       # oracle-vs-noop calibration
python -m bench run             # the oracle across the full matrix
python -m bench gradecompare    # cheap graders (DOM/receipt/screenshot) scored vs SQL
python -m bench hardened        # naive vs hardened recovery under faults
python -m bench report          # write the static HTML report to runs/
python -m bench customer        # calibrate the customer-config layer example
python -m bench eval ...        # run an agent + gate on the result (see below)
```

Each still works standalone too (`python -m bench.selfcheck`, etc.).

## Evaluating an agent — `python -m bench eval`

Runs an agent across a condition set and returns a CI exit code: **0** when it clears the
gate, **1** when it doesn't.

| Flag | Default | Meaning |
|---|---|---|
| `--agent` | `oracle` | agent spec (built-in or `module:factory` — see below) |
| `--model` | – | model id for the `llm` agent |
| `--conditions` | `seven` | `clean` \| `seven` (clean + 6 UI mutations) \| `faults` (clean + 3 infra faults) |
| `--trials` | `3` | trials per cell |
| `--tasks` | all | comma-separated task ids |
| `--report` | – | write the HTML report to this path |
| `--min-pass-rate` | – | fail if `passed/planned` is below this (0..1) |
| `--allow-skips` | off | don't fail the build on infra-skipped cells |
| `--json` | off | machine-readable summary |

```bash
# Gate: the oracle must sweep the matrix (harness sanity check).
python -m bench eval --agent oracle --conditions seven --trials 1 --min-pass-rate 1.0

# Measure the hardened agent under faults, write a report, machine-readable:
python -m bench eval --agent hardened --conditions faults --report runs/report.html --json
```

The gate is explicit, not magic: `--min-pass-rate` thresholds `passed/planned`, and an
infra-skipped cell (the agent never got a fair shot) fails the build unless `--allow-skips`.

## Bring your own agent

`--agent` accepts:

- **a built-in** — `oracle` (scripted perfect run), `hardened` (the A2 recovery agent), or
  `llm` (the real `LLMPolicy`; needs `ANTHROPIC_API_KEY` and the `llm` extra).
- **a `module:factory` entrypoint** — your own code. `factory()` must return a fresh
  [`RunFn`](../bench/adapter.py): an async `(case, condition, trial) -> Transcript` that drives
  your agent and leaves its result in the case's database.

If your agent is a `Policy` (speaks `decide(instruction, observation, history) -> Action`),
wrap it in one line with the exported helper:

```python
# mypkg/agent.py
from bench.adapter import policy_run_fn

def make_policy(case):
    return MyPolicy(...)          # fresh instance per case

RUN = policy_run_fn(make_policy)  # a RunFn
```

```bash
pip install -e .                  # make mypkg importable
python -m bench eval --agent mypkg.agent:RUN --conditions seven
```

## GitHub Action

This repo's own CI (`.github/workflows/ci.yml`) installs, runs the tests, `selfcheck`, the
customer calibration, then the oracle eval gate, and uploads `runs/report.html` as an artifact.

To gate **your** agent in CI, use the reusable composite action:

```yaml
jobs:
  assay:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e .            # install your agent package (exposes mypkg.agent:RUN)
      - uses: DhruvDS2/assay@main
        with:
          agent: "mypkg.agent:RUN"
          conditions: "seven"
          trials: "3"
          min-pass-rate: "0.8"
```

The action installs Assay from its own checkout and runs the eval gate; the build fails when
your agent drops below `min-pass-rate`. See [`action.yml`](../action.yml) for all inputs.
