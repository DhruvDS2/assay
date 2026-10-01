# Assay

**Assay** — the trust layer for web agents: a deployable agent plus an external, ground-truth
check that *certifies* whether it actually did the job (to *assay* is to test something for its
true quality). "Did the agent succeed?" is a **SQL query against a SQLite database**, not an
LLM's guess about a screenshot.

> `groundtruth` is the internal codename — the repo dir, the `app/`+`bench/` modules, the env,
> and the memory graph all keep that name; **Assay** is the product/brand on top.

The core bet: own the website, so we can (1) grade by reading the database, (2) reset to an
identical world per case, and (3) mutate the site on purpose to measure what breaks the agent.
Primary customer: companies running agents on their *own* stores — they own the DB, so the
ground-truth check works for real on their site.

## Architecture

- **`app/`** — the shop (the environment under test). ~300 lines of FastAPI + SQLite, plain
  HTML forms, **no JavaScript**. The page the agent sees is exactly what the server sent; every
  action is a form POST that lands in SQLite, so nothing happens the grader can't see.
  - `main.py` — routes: login, browse, cart, checkout, orders, cancel
  - `db.py` — connection, password hashing, seed data
  - `schema.sql` — 5 tables: `users`, `products`, `cart_items`, `orders`, `order_items`
  - `templates/` — Jinja2 HTML
- **`bench/`** — the instrument
  - `model.py` — core types: `Task`, `Check`, `Baseline`, `Case`, `Grade`
  - `env.py` — seed once, copy per case, capture baseline
  - `tasks.py` — the 5 v0 tasks (setup, checks, oracle)
  - `grader.py` — runs every check against a case's database
  - `selfcheck.py` — calibration CLI
- **`tests/`** — `test_app.py` (the shop behaves) and `test_ground_truth.py` (the grader tells the truth)

## Key concepts (map 1:1 to `bench/model.py`)

- **Task** — a goal in plain English + start state + list of checks + an oracle. Self-contained.
- **Check** — one named yes/no question about the DB (`correct_items`, `no_new_orders`). A task
  passes only if **all** its checks pass. Named checks make failures explainable, not just counted.
- **Baseline** — a snapshot taken after setup, before the agent runs. Separates what the agent
  did from what was already there → makes side-effect checks ("exactly one new order") possible.
- **Oracle** — a scripted correct run through the same forms a browser uses. Proves the task is achievable.
- **Case** — one task on one fresh database. The unit of isolation.
- **Grade** — the result of every check for one case.

## The v0 tasks

| Task | Agent must | The trap |
|---|---|---|
| `buy_blue_shirt` | Buy one Blue Oxford Shirt to a given address | A White Oxford Shirt exists at the same price |
| `buy_socks_x3` | Buy 3 Wool Socks in one order | Placing three separate orders of one |
| `buy_tote_and_mug` | Buy a tote and a mug, nothing else | Cart already holds a Black Cap it must remove |
| `cancel_latest_order` | Cancel the most recent order only | Two orders exist; the older is listed too |
| `empty_cart` | Empty the cart without buying | Checking out also empties the cart |

`empty_cart` is the argument for multiple named checks: buying everything ends with an empty cart,
so `cart_empty` alone passes it — `no_new_orders` catches it. `new_orders` counts orders in **any**
status, so double-order-then-cancel still fails `exactly_one_new_order`.

## Workflow

Every case: `seed.db` → file-copy to `case.db` → `task.setup` → capture baseline → agent runs
through real web forms → grader runs SQL checks → per-check pass/fail.

**Calibration** — before any agent result means anything, the grader must prove it separates
success from failure. Every task ships an **oracle** (perfect run, must pass all checks) and is
tested against a **noop** agent (does nothing, must fail some check).

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # add ".[solari]" for v1

pytest                        # 25 tests: app, calibration, adversarial agents
python -m bench.selfcheck     # oracle vs noop on every task; must print "calibrated"

# Run the shop yourself:
uvicorn app.main:default_app --factory --reload
# http://127.0.0.1:8000 — login alice@example.com/password123 or bob@example.com/hunter22
```

Run `python -m bench.selfcheck` after changing any task or the app. It exits non-zero if a task
is broken (oracle fails a check) or the checks are too weak (noop passes everything).

The DB lives at `data/app.db` (delete to reset; override with `GT_DB_PATH`). `data/`, `runs/`,
`*.db`, and `.env` are gitignored.

## Adding a task

1. Write `setup(conn)` in `bench/tasks.py` (or reuse `_no_setup`).
2. Write checks — at least one that **doing nothing fails**, and one **side-effect check**
   (`no_new_orders` or `exactly_one_new_order`).
3. Write the oracle through the same endpoints a browser uses.
4. Add it to `TASKS`.
5. Run `python -m bench.selfcheck` — if it doesn't say `calibrated`, the task isn't done.
6. Add one adversarial test to `tests/test_ground_truth.py`: the most plausible wrong agent and
   the exact check it should fail. **This is the step people skip and the one that matters most** —
   if you can't name the check a wrong agent fails, you don't know what the task measures.

## Design decisions (don't relitigate)

- **Own the app instead of testing real sites** — gives up realism, gets truth/reproducibility/control.
- **No JavaScript in the shop** — the agent sees exactly what the server sent. JS mutations come later as perturbations.
- **One SQLite file per case, copied from a seed** — isolation for the price of a file copy; safe for parallel cases.
- **Checks are named and granular**, and **only judge what the agent controls** (no order-total checks — that's the app's job).
- **Every task has an oracle** — an unachievable task and a weak agent look identical without it.
- **Money in integer cents** — no float rounding, ever.
- **Credentials go in the instruction** — the agent logs in itself, like a real user.

## Roadmap

**North star — the trust layer for web agents.** Two things that compound: **the Agent**
(a deployable shopping agent — the *doer*) and **the Eval** (an external, ground-truth check
that *certifies* it — the agent never grades its own homework). **Primary customer: Product 2 —
companies testing/running agents on their *own* stores.** They own their database, so the
SQL-truth premise works for real on their site; the eval is the proof that makes the agent sellable.

Research benchmark → product. Think in three tracks, built in this order: **A2 → A3 → all of B → all of C.**

### Foundations — DONE
v0 (shop, seed, 5 tasks, SQL grader, calibration, adversarial tests) · v1 (`bench/engine.py` async DAG,
`bench/run.py` matrix, `bench/agent.py` observe→decide→act with `Browser`/`Policy`, honest metrics:
pass^k, passed/attempted vs passed/planned, skipped separately) · `app/mutations.py` (6 UI mutations →
7-condition matrix, 5×7×3 = 105 cases) · v2 graders (`bench/graders.py`: SQL vs DOM vs LLM-screenshot +
false-success rate) · v3 faults (`app/faults.py`: error_on_checkout, drop_checkout_response, expire_session;
fault axis wired into the matrix; recovery double-order rate) · `LLMPolicy` seam + localhost condition viewer.

Core invariants (still hold everywhere): **retry infra errors, never agent mistakes** (a finished-but-wrong
agent returns, only infra raises → engine retry); **every retry starts from a fresh DB**; **grade nodes are
pure** (read a local file, no network).

### Track A — The Agent (the doer)
- **A1 · Go live** (enabling gate, whenever the key is added) — `ANTHROPIC_API_KEY` → run the real `LLMPolicy`
  across the 105-case matrix → first real robustness numbers. A2/A3 can be *built* keyless (scripted/fake
  stand-ins) but need A1 to be *measured*.
- **A2 · Harden the agent** — add the loop capabilities it lacks today: planning, self-verification, recovery.
  Target: higher pass^k under mutations/faults. (NEXT.)
- **A3 · Site knowledge-graph memory** (old v5) — the agent builds/consults a graph of the site
  (pages → elements → actions → outcomes) before acting and updates it after. Measure: does it help on a clean
  site and *hurt* once the site is mutated (stale graph)? Needs the A1 no-memory baseline.

### Track B — The Eval / Trust layer (the judge)
- **B1 · Report** (old v4) — static HTML matrix report (worst cells first, per-failure trace + rrweb replay).
  This is the sales-proof artifact.
- **B2 · Generalize off our shop** — the core of Product 2: a config where a *customer* declares their own
  tasks + SQL checks against *their* schema, and seeds/resets *their* DB. The harness stops being welded to
  our shop.
- **B3 · Production graders** — turn v2's false-success numbers into calibrated confidence bands; add
  receipt/API graders and a human-review queue for the uncertain tail.

### Track C — Packaging
- **C1 · Bring-your-own-agent adapter + CLI / GitHub Action** — the eval/CI harness surface.
- **C2 · Sandbox/hosting (Solari direction)** — run the customer's site + agent in isolated VMs, pull the DB
  back to grade; scales B2. See Solari notes below.
- **C3 · Multi-tenancy, auth, dashboards.**

Product map: **Product 2 (QA on owned site) = B2 + C2** (primary) · Product 1 (CI/eval harness) = core + C1 + B1 ·
Product 3 (prod monitor) = B3.

## Solari integration (v1)

Runs on [Solari](https://getsolari.com): the shop runs in a Solari sandbox (microVM from a snapshot),
the agent drives a Solari cloud browser. The grader doesn't change — it still reads a SQLite file
pulled back from the VM. `SOLARI_API_KEY` and `GT_SECRET_KEY` go in `.env` (never commit). Docs:
[docs.getsolari.com](https://docs.getsolari.com). Gotchas that shape the design:

- **`kill()` ends a VM; `close()` doesn't** — `kill()` goes in `finally`. `browser.close()` releases the session — also in `finally`.
- **`timeout_ms` is a rolling idle window**, reset on every use — not a hard deadline. The engine's per-node timeout is the real limit.
- **Commands are not shell-interpreted** — program in `cmd`, argv in `args`.
- **Recording is opt-in** (`recording=True`); replays upload after release, so the first download usually 404s — poll ~30s.
- **Concurrency is plan-limited** → becomes the engine's `group="solari"` cap.
- **Preview URLs are public** — set `GT_SECRET_KEY` per run rather than the dev default.
- `Sandbox.revert(snapshot_id)` exists but is undocumented — verify its semantics before designing around VM reuse.

## Memory: use the knowledge graph

Two distinct knowledge graphs live in this project — don't conflate them:

1. **The benchmarked agent's memory (v5, product feature)** — a graph of the *site* (pages,
   elements, actions, outcomes: "*Place order* on `/checkout` creates an order") that the agent
   consults before acting and updates after. The v5 experiment measures whether such memory helps
   on a clean site and *hurts* once the site is mutated (stale graph). Meaningless without the v1
   no-memory baseline, so it comes last.

2. **My working memory across sessions on this repo** — stored as linked files under
   `~/.claude/projects/-Users-dhruvsharma-Desktop-groundtruth/memory/`, forming a knowledge graph:
   each memory file is a **node**, and `[[wikilink]]` references between them are **edges**.
   Prefer linking related memories over writing standalone facts, so the memory is a navigable
   graph rather than a flat list. `MEMORY.md` is the index. See that directory for the current graph.
