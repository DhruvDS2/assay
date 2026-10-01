"""
v2: graders that are NOT the truth — so we can measure how often they lie.

The SQL grader (bench/grader.py) is ground truth: it reads what actually happened.
Real-world agent benchmarks rarely have that luxury — they grade from what's
*shown*: a DOM scrape, or an LLM judging a screenshot. Because we own the app,
we can run those weaker graders side-by-side with SQL and score them against it.

Three graders, same yes/no question ("did the task succeed?"):

- `SqlGrader`      — the truth. Reads the database.
- `DomGrader`      — a shallow read of the final rendered pages (the kind of
                     keyword/structure check a DOM-scraping harness would do).
- `LlmScreenshotGrader` — an LLM looks at the final page text (our no-JavaScript
                     proxy for a screenshot) plus the goal, and rules pass/fail.

`grade_all` renders the final screens once and runs every grader on them, then
`false_success_rate` scores each non-SQL grader against SQL. The headline number
is the screenshot judge's false-success rate: it said "done" when SQL says it
wasn't — the error that silently inflates agent scores everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from bench.agent import visible_text
from bench.grader import grade
from bench.model import Case


@dataclass(frozen=True)
class Verdict:
    """One grader's call on one case."""

    grader: str
    passed: bool
    detail: str = ""


class Grader(Protocol):
    name: str

    def judge(self, case: Case, state: dict[str, str]) -> Verdict: ...


# --------------------------------------------------------------- rendered screens

def render_state(case: Case, mutations: tuple[str, ...] = ()) -> dict[str, str]:
    """
    The final screens a display-based grader would see: log in as the task's user
    and read the pages where success shows up (storefront, cart, orders, and the
    latest order's detail), as *visible text* — what a screenshot would contain.

    `mutations` serves the same perturbed app the agent saw, so a screenshot judge
    is scored on the page it would actually have been handed, not a clean one.
    """
    import re

    from fastapi.testclient import TestClient

    from app.mutations import create_mutated_app

    state: dict[str, str] = {}
    with TestClient(create_mutated_app(case.db_path, mutations=mutations)) as c:
        c.post("/login", data={"email": case.task.email, "password": case.task.password})
        for path in ("/", "/cart", "/orders"):
            state[path] = visible_text(c.get(path).text)
        orders_html = c.get("/orders").text
        m = re.search(r'href="(/orders/\d+)"', orders_html)   # newest order, if any
        if m:
            state["/orders/latest"] = visible_text(c.get(m.group(1)).text)
    return state


# ------------------------------------------------------------------- SQL (truth)

class SqlGrader:
    name = "sql"

    def judge(self, case: Case, state: dict[str, str]) -> Verdict:
        g = grade(case)
        failed = [r.name for r in g.results if not r.passed]
        detail = "all checks pass" if g.passed else "failed: " + ", ".join(failed)
        return Verdict(self.name, g.passed, detail)


# ---------------------------------------------------------------------- DOM grader
# Shallow per-task reads of the rendered pages — honest about being shallow. Each
# returns (passed, why). They mostly agree with SQL on a clean site, and diverge
# when a mutation changes what's *shown* without changing what *happened*.


def _dom_buy_blue_shirt(s):
    d = s.get("/orders/latest", "")
    ok = "blue oxford shirt" in d and "12 elm" in d and "placed" in d
    return ok, "order page shows blue shirt shipped to 12 Elm" if ok else "blue shirt / address not shown"


def _dom_buy_socks_x3(s):
    d = s.get("/orders/latest", "")
    ok = "wool socks" in d and "3" in d and "placed" in d
    return ok, "order page shows 3 wool socks" if ok else "3 wool socks not shown"


def _dom_buy_tote_and_mug(s):
    d = s.get("/orders/latest", "")
    ok = "canvas tote" in d and "enamel mug" in d and "black cap" not in d
    return ok, "order page shows tote + mug, no cap" if ok else "tote/mug/cap wrong on order page"


def _dom_cancel_latest(s):
    o = s.get("/orders", "")
    ok = "cancelled" in o and "placed" in o
    return ok, "orders list shows one cancelled, one placed" if ok else "cancelled/placed not both shown"


def _dom_empty_cart(s):
    ok = "cart is empty" in s.get("/cart", "")
    return ok, "cart page says it's empty" if ok else "cart not shown empty"


_DOM_CHECKS: dict[str, Callable[[dict[str, str]], tuple[bool, str]]] = {
    "buy_blue_shirt": _dom_buy_blue_shirt,
    "buy_socks_x3": _dom_buy_socks_x3,
    "buy_tote_and_mug": _dom_buy_tote_and_mug,
    "cancel_latest_order": _dom_cancel_latest,
    "empty_cart": _dom_empty_cart,
}


class DomGrader:
    name = "dom"

    def judge(self, case: Case, state: dict[str, str]) -> Verdict:
        check = _DOM_CHECKS.get(case.task.id)
        if check is None:
            return Verdict(self.name, False, "no DOM check for this task")
        lower = {k: v.lower() for k, v in state.items()}
        ok, why = check(lower)
        return Verdict(self.name, ok, why)


# ----------------------------------------------------------- LLM screenshot grader

_JUDGE_SYSTEM = """\
You are grading whether a web agent accomplished its goal, judging ONLY from what \
the final screens show — as a human reviewer glancing at screenshots would. You \
cannot see any database. Decide from the page text alone, then call `verdict`."""

_VERDICT_TOOL = {
    "name": "verdict",
    "description": "Report whether the goal was accomplished, per the screens.",
    "input_schema": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "reason": {"type": "string", "description": "One sentence, citing the screen."},
        },
        "required": ["success", "reason"],
        "additionalProperties": False,
    },
}


class LlmScreenshotGrader:
    """An LLM ruling on success from the rendered screens — the judge we want to catch lying."""

    name = "screenshot"

    def __init__(self, *, model: str = "claude-opus-4-8", client: object = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def judge(self, case: Case, state: dict[str, str]) -> Verdict:
        screens = "\n\n".join(f"[screen {p}]\n{t}" for p, t in state.items())
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=_JUDGE_SYSTEM,
            tools=[_VERDICT_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content":
                       f"GOAL: {case.task.goal}\n\nFINAL SCREENS:\n{screens}\n\n"
                       "Did the agent accomplish the goal?"}],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "verdict":
                inp = block.input
                return Verdict(self.name, bool(inp.get("success")), inp.get("reason", ""))
        return Verdict(self.name, False, "judge returned no verdict")


# --------------------------------------------------------------- run & score them

def grade_all(case: Case, graders, mutations: tuple[str, ...] = ()) -> list[Verdict]:
    """Render the final screens once; run every grader against the same case + screens."""
    state = render_state(case, mutations=mutations)
    return [g.judge(case, state) for g in graders]


@dataclass(frozen=True)
class GraderScore:
    """How one non-SQL grader compares to SQL truth across many cases."""

    grader: str
    total: int
    agree: int
    false_success: int        # grader said PASS, SQL said FAIL — the dangerous one
    false_failure: int        # grader said FAIL, SQL said PASS

    @property
    def false_success_rate(self) -> float:
        return self.false_success / self.total if self.total else 0.0


def false_success_rate(rows: list[list[Verdict]]) -> dict[str, GraderScore]:
    """
    `rows` is one `grade_all` result per case. Score every non-SQL grader against
    the SQL verdict in the same row. This is the v2 deliverable: a disagreement
    table whose headline cell is each judge's false-success rate.
    """
    scores: dict[str, dict[str, int]] = {}
    for row in rows:
        by_name = {v.grader: v for v in row}
        truth = by_name.get("sql")
        if truth is None:
            continue
        for v in row:
            if v.grader == "sql":
                continue
            s = scores.setdefault(v.grader, {"total": 0, "agree": 0, "fs": 0, "ff": 0})
            s["total"] += 1
            if v.passed == truth.passed:
                s["agree"] += 1
            elif v.passed and not truth.passed:
                s["fs"] += 1
            else:
                s["ff"] += 1
    return {
        name: GraderScore(name, s["total"], s["agree"], s["fs"], s["ff"])
        for name, s in scores.items()
    }
