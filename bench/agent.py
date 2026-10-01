"""
The agent harness: drive a web agent through a browser and record what it did.

    run(instruction, browser, policy) -> Transcript

`run` is the observe → decide → act loop. It knows nothing about *which* agent
is deciding (that's the injected `Policy`) or *how* the browser reaches the site
(that's the `Browser`). It just runs the loop honestly and writes down every
step, so a grader can later read the truth from SQL and a replay can show the
run frame by frame.

The `Browser` protocol is the subset of a Playwright `Page` the loop uses —
`goto`, `content`, `fill`, `click`, `url`. A real Playwright/Solari page
satisfies it as-is; `TestClientBrowser` below satisfies it offline by parsing
the shop's plain HTML forms and POSTing them, so the whole harness is testable
without a cloud browser.

The one invariant that ties this to the rest of the bench:

    An agent *mistake* (bad selector, missing field) is recorded in the
    transcript and the loop continues — it is never raised. Only an
    *infrastructure* failure (the browser/transport dying) raises out of `run`,
    where bench/run.py turns it into a retry. Retrying a wrong answer would hide
    the flakiness the bench exists to measure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable
from urllib.parse import urlsplit


# --------------------------------------------------------------------- vocabulary


@dataclass(frozen=True)
class Action:
    """One thing the agent decided to do."""

    kind: str                 # "goto" | "fill" | "click" | "stop"
    target: str = ""          # url (goto), selector (fill/click)
    value: str = ""           # text to type (fill)
    note: str = ""            # free-text reason, esp. for "stop"

    @classmethod
    def goto(cls, url: str) -> "Action":
        return cls("goto", target=url)

    @classmethod
    def fill(cls, selector: str, value: str) -> "Action":
        return cls("fill", target=selector, value=value)

    @classmethod
    def click(cls, selector: str) -> "Action":
        return cls("click", target=selector)

    @classmethod
    def stop(cls, note: str = "") -> "Action":
        return cls("stop", note=note)


@dataclass(frozen=True)
class Observation:
    """What the agent sees before it decides: the page, as URL + HTML + text."""

    url: str
    html: str
    text: str


@dataclass(frozen=True)
class Step:
    index: int
    observation: Observation
    action: Action
    error: Optional[str] = None      # set iff the action was a recorded mistake


@dataclass(frozen=True)
class Transcript:
    """The full record of one agent run — the raw material for grading & replay."""

    instruction: str
    steps: tuple[Step, ...] = field(default_factory=tuple)
    stopped_reason: str = ""          # "stopped" | "max_steps" | agent's own note
    final_url: str = ""

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def mistakes(self) -> list[Step]:
        return [s for s in self.steps if s.error is not None]


class AgentActionError(Exception):
    """
    The agent asked for something the page can't do (no such button, no such
    field). A *mistake*, not an infra failure: the loop records it and moves on.
    """


# --------------------------------------------------------------------- protocols


@runtime_checkable
class Browser(Protocol):
    """The slice of a Playwright `Page` the loop needs. Async, like Playwright."""

    @property
    def url(self) -> str: ...

    async def goto(self, url: str) -> None: ...

    async def content(self) -> str: ...

    async def fill(self, selector: str, value: str) -> None: ...

    async def click(self, selector: str) -> None: ...


class Policy(Protocol):
    """The brain. Given what it sees, decide the next action."""

    def decide(
        self, instruction: str, observation: Observation, history: Sequence[Step]
    ) -> Action: ...


# ------------------------------------------------------------------------ the loop


async def _observe(browser: Browser) -> Observation:
    html = await browser.content()
    return Observation(url=browser.url, html=html, text=visible_text(html))


async def _execute(browser: Browser, action: Action) -> None:
    if action.kind == "goto":
        await browser.goto(action.target)
    elif action.kind == "fill":
        await browser.fill(action.target, action.value)
    elif action.kind == "click":
        await browser.click(action.target)
    else:
        raise AgentActionError(f"unknown action kind {action.kind!r}")


async def run(
    instruction: str,
    browser: Browser,
    policy: Policy,
    *,
    max_steps: int = 25,
) -> Transcript:
    """
    Run the agent to completion and return its transcript.

    Completion means the policy emits `stop`, or `max_steps` is reached. Either
    way this returns normally — a finished agent, right or wrong, is not an
    error. Agent mistakes are recorded on their step; only infra failures
    (anything other than `AgentActionError`) propagate.
    """
    steps: list[Step] = []
    reason = "max_steps"
    for i in range(max_steps):
        obs = await _observe(browser)
        action = policy.decide(instruction, obs, tuple(steps))
        if action.kind == "stop":
            steps.append(Step(i, obs, action))
            reason = action.note or "stopped"
            break
        error: Optional[str] = None
        try:
            await _execute(browser, action)
        except AgentActionError as e:   # a mistake — record it, keep going
            error = str(e)
        steps.append(Step(i, obs, action, error))
    return Transcript(
        instruction=instruction,
        steps=tuple(steps),
        stopped_reason=reason,
        final_url=browser.url,
    )


# ------------------------------------------------------------------ scripted brain


ScriptStep = "Action | Callable[[Observation], Action]"


class ScriptedPolicy:
    """
    A deterministic policy that plays a fixed list of steps. Each step is either
    a literal `Action` or a callable `(observation) -> Action` for the one bit
    that must read the page (e.g. finding the latest order's link). The honest
    stand-in that lets us test the whole harness without an LLM — the browser
    analogue of the oracle.
    """

    def __init__(self, steps: Sequence[ScriptStep]) -> None:
        self._steps = list(steps)
        self._i = 0

    def decide(self, instruction, observation, history) -> Action:
        if self._i >= len(self._steps):
            return Action.stop("script exhausted")
        step = self._steps[self._i]
        self._i += 1
        return step(observation) if callable(step) else step


# ---------------------------------------------------------- offline HTML browser
# A tiny form-and-link browser so the harness runs without a cloud browser. The
# shop is plain HTML with no JavaScript, so "click a submit button" is just
# "POST the enclosing form" and "click a link" is "GET its href".


@dataclass
class _Form:
    action: str
    fields: dict[str, str]              # name -> default value (hidden, number, textarea)
    ids: dict[str, str]                # id  -> name (so #id selectors resolve)
    buttons: list[str]                 # visible submit-button labels


@dataclass
class _Link:
    text: str
    href: str


class _PageModel(HTMLParser):
    """Parse a page into its forms and links. Just enough HTML for this shop."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[_Form] = []
        self.links: list[_Link] = []
        self._form: Optional[_Form] = None
        self._capture: Optional[list[str]] = None   # accumulates button/link text
        self._pending_href: Optional[str] = None

    @classmethod
    def parse(cls, html: str) -> "_PageModel":
        p = cls()
        p.feed(html)
        return p

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form":
            self._form = _Form(action=a.get("action", ""), fields={}, ids={}, buttons=[])
            self.forms.append(self._form)
        elif tag in ("input", "textarea") and self._form is not None:
            name = a.get("name")
            if name:
                self._form.fields.setdefault(name, a.get("value", ""))
                if a.get("id"):
                    self._form.ids[a["id"]] = name
            itype = a.get("type", "")
            if tag == "input" and itype in ("submit", "button", "image"):
                self._form.buttons.append(a.get("value", ""))
            if tag == "textarea":
                self._capture = []          # textarea body is its default value
        elif tag == "button" and self._form is not None:
            self._capture = []
        elif tag == "a":
            self._pending_href = a.get("href", "")
            self._capture = []

    def handle_data(self, data):
        if self._capture is not None:
            self._capture.append(data)

    def handle_endtag(self, tag):
        if tag == "form":
            self._form = None
        elif tag == "textarea" and self._form is not None:
            self._capture = None            # value stays as parsed default ("")
        elif tag == "button" and self._form is not None:
            self._form.buttons.append("".join(self._capture or []).strip())
            self._capture = None
        elif tag == "a":
            if self._pending_href is not None:
                self.links.append(_Link("".join(self._capture or []).strip(), self._pending_href))
            self._pending_href = None
            self._capture = None


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def visible_text(html: str) -> str:
    """Cheap tag-strip for readable transcripts and (later) LLM prompts."""
    return _WS_RE.sub(" ", unescape(_TAG_RE.sub(" ", html))).strip()


def _attr(selector: str, name: str) -> Optional[str]:
    m = re.search(rf"{name}=['\"]?([^'\"\]]+)", selector)
    return m.group(1) if m else None


class TestClientBrowser:
    """
    A `Browser` backed by Starlette's TestClient — the offline transport.

    It follows redirects and keeps cookies across requests (so login sticks),
    exactly like a real browser session, but never leaves the process. Swap it
    for a Solari/Playwright page and the agent loop is unchanged.
    """

    __test__ = False   # its name starts with "Test"; keep pytest from collecting it

    def __init__(self, app) -> None:
        from fastapi.testclient import TestClient

        self._client = TestClient(app)
        self._url = ""
        self._html = ""
        self._pending: dict[str, str] = {}

    # --- Browser protocol ---------------------------------------------------

    @property
    def url(self) -> str:
        return self._url

    async def content(self) -> str:
        return self._html

    async def goto(self, url: str) -> None:
        self._get(url)

    async def fill(self, selector: str, value: str) -> None:
        page = _PageModel.parse(self._html)
        name = self._resolve_field(page, selector)
        if name is None:
            raise AgentActionError(f"no fillable field for selector {selector!r}")
        self._pending[name] = value

    async def click(self, selector: str) -> None:
        page = _PageModel.parse(self._html)

        # An explicit href selector, or a bare path, is a link click (GET).
        href = _attr(selector, "href")
        if href is None and selector.strip().startswith("/"):
            href = selector.strip()
        if href is not None:
            self._get(href)
            return

        # text=… matches a link first, then a submit button (Playwright-ish).
        if selector.startswith("text="):
            want = selector[len("text="):].strip().strip("\"'").lower()
            for lk in page.links:
                if want in lk.text.lower():
                    self._get(lk.href)
                    return
            for form in page.forms:
                if any(want in b.lower() for b in form.buttons):
                    self._submit(form)
                    return
            raise AgentActionError(f"no link or button matching {selector!r}")

        # form[action='…'] submits that form.
        action = _attr(selector, "action")
        if action is not None:
            for form in page.forms:
                if form.action == action:
                    self._submit(form)
                    return
            raise AgentActionError(f"no form with action {action!r}")

        # A bare button click only works when the page has exactly one form.
        if "button" in selector:
            if len(page.forms) == 1:
                self._submit(page.forms[0])
                return
            raise AgentActionError("ambiguous button click: page has multiple forms")

        raise AgentActionError(f"unsupported click selector {selector!r}")

    # --- internals ----------------------------------------------------------

    def _resolve_field(self, page: _PageModel, selector: str) -> Optional[str]:
        s = selector.strip()
        if s.startswith("#"):
            fid = s[1:]
            for f in page.forms:
                if fid in f.ids:
                    return f.ids[fid]
            return None
        name = _attr(s, "name")
        if name is not None:
            return name if any(name in f.fields for f in page.forms) else None
        for f in page.forms:            # bare token: try name, then id
            if s in f.fields:
                return s
        for f in page.forms:
            if s in f.ids:
                return f.ids[s]
        return None

    def _submit(self, form: _Form) -> None:
        data = dict(form.fields)
        for k in list(data):
            if k in self._pending:
                data[k] = self._pending[k]
        resp = self._client.post(form.action, data=data, follow_redirects=True)
        self._set_state(resp)

    def _get(self, url: str) -> None:
        resp = self._client.get(self._rel(url), follow_redirects=True)
        self._set_state(resp)

    def _set_state(self, resp) -> None:
        self._url = str(resp.url)
        self._html = resp.text
        self._pending = {}

    @staticmethod
    def _rel(url: str) -> str:
        parts = urlsplit(url)
        path = parts.path or "/"
        return f"{path}?{parts.query}" if parts.query else path


# ------------------------------------------------------------- run.py integration


def testclient_run_fn(policy_for: Callable[[object], Policy], *, max_steps: int = 25):
    """
    Build a bench/run.py `RunFn` that drives `agent.run` over a TestClientBrowser.

    `policy_for(case) -> Policy` yields a fresh policy per case (scripts depend
    on the task). Async, so the engine awaits it: any transport error raised
    here propagates as an infra failure and is retried; a finished agent returns.
    """
    from app.main import create_app

    async def run_fn(case, condition, trial) -> Transcript:
        browser = TestClientBrowser(create_app(case.db_path))
        instruction = case.task.instruction("http://shop")
        return await run(instruction, browser, policy_for(case), max_steps=max_steps)

    return run_fn


testclient_run_fn.__test__ = False   # name starts with "test"; not a pytest test


# ---------------------------------------------------------------- the LLM brain
# The first real `Policy`: an LLM decides each action from what it sees. Everything
# above this line runs offline; this is the only part that needs a network and an
# API key. It speaks the same `decide` contract as `ScriptedPolicy`, so the loop,
# the browser, and the grader are all unchanged — only the brain is new.


# The selectors below are the subset both TestClientBrowser and a real Playwright
# page understand, so the same policy drives the offline harness and the cloud
# browser without change. We teach the model exactly this vocabulary.
_LLM_SYSTEM = """\
You are a careful web-shopping agent driving a small online store through its \
plain HTML forms. Each turn you see the current page (URL, visible text, and the \
exact links, buttons, and form fields on it) and the history of what you've done. \
Decide the single next action by calling the `act` tool.

Action kinds:
- "goto": navigate to a path. Put the path (e.g. "/cart", "/orders") in `target`.
- "click": follow a link or submit a button. Put `text=<the visible label>` in \
`target` — e.g. `text=Add to cart`. Match the label you actually see on the page.
- "fill": type into a form field. Put the field selector in `target` — either \
`#<id>` or `name=<name>` from the field list — and the text in `value`. Filling \
does not submit; click the button afterwards.
- "stop": you are done (or stuck). Explain why in `note`.

Rules:
- Do exactly what the task asks and nothing more. Side effects are graded: an \
extra order, an item left in the cart, or the wrong item all count as failures.
- Read the labels on THIS page; they may be worded unusually. Pick the control \
whose meaning matches your intent, not a label you expected.
- Think about quantities and which specific product variant the task names.
- When the task is fully accomplished, "stop". Don't keep acting."""


_ACT_TOOL = {
    "name": "act",
    "description": "Take exactly one action in the browser.",
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["goto", "fill", "click", "stop"]},
            "target": {
                "type": "string",
                "description": "goto: a path like /cart. click: text=<label>. "
                "fill: #<id> or name=<name>. stop: empty.",
            },
            "value": {
                "type": "string",
                "description": "fill: the text to type. Otherwise empty.",
            },
            "note": {
                "type": "string",
                "description": "A brief reason for the action; for stop, why you're done.",
            },
        },
        "required": ["kind", "target", "value", "note"],
        "additionalProperties": False,
    },
}


def _affordances(html: str) -> str:
    """Render a page's links, buttons, and fields — the things an agent can act on."""
    page = _PageModel.parse(html)
    lines: list[str] = []
    if page.links:
        lines.append("Links (click with text=<label>, or goto the path):")
        for lk in page.links:
            lines.append(f"  - {lk.text!r} -> {lk.href}")
    for i, form in enumerate(page.forms):
        lines.append(f"Form -> {form.action}")
        for name, default in form.fields.items():
            ids = [fid for fid, nm in form.ids.items() if nm == name]
            sel = f"#{ids[0]}" if ids else f"name={name}"
            lines.append(f"    field {sel} (name={name!r}, current={default!r})")
        for b in form.buttons:
            if b:
                lines.append(f"    button {b!r} (click with text={b})")
    return "\n".join(lines) if lines else "(no actionable elements on this page)"


class LLMPolicy:
    """
    A `Policy` whose brain is a Claude model. Given the page and the history, it
    calls the `act` tool to choose one action — the real agent the bench exists
    to measure, in place of the scripted stand-in.

    Transport errors (network, 5xx after the SDK's own retries) propagate out of
    `decide` -> `run` -> run.py as infra failures to be retried. A model that
    answers but answers badly returns a normal action: a wrong answer is the
    flakiness we measure, never a retry.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-8",
        max_history: int = 8,
        max_tokens: int = 1024,
        client: object = None,
    ) -> None:
        if client is None:
            import anthropic  # optional dep; only needed for a real agent run

            client = anthropic.Anthropic()
        self._client = client
        self._model = model
        self._max_history = max_history
        self._max_tokens = max_tokens

    def decide(self, instruction, observation, history) -> Action:
        prompt = self._prompt(instruction, observation, history)
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_LLM_SYSTEM,
            tools=[_ACT_TOOL],
            tool_choice={"type": "any"},     # it must act; `act` is the only tool
            messages=[{"role": "user", "content": prompt}],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "act":
                return self._to_action(block.input)
        # No tool call (e.g. a refusal): a finished-but-unhelpful turn, not infra.
        return Action.stop("model returned no action")

    def _prompt(self, instruction, observation, history) -> str:
        recent = history[-self._max_history:]
        hist = "\n".join(
            f"  {s.index}: {s.action.kind} {s.action.target!r} {s.action.value!r}"
            + (f"  -> ERROR: {s.error}" if s.error else "")
            for s in recent
        ) or "  (nothing yet)"
        return (
            f"TASK: {instruction}\n\n"
            f"CURRENT URL: {observation.url}\n\n"
            f"PAGE TEXT:\n{observation.text}\n\n"
            f"ACTIONABLE ELEMENTS:\n{_affordances(observation.html)}\n\n"
            f"HISTORY:\n{hist}\n\n"
            "Choose the next action."
        )

    @staticmethod
    def _to_action(inp: dict) -> Action:
        kind = inp.get("kind", "stop")
        target, value, note = inp.get("target", ""), inp.get("value", ""), inp.get("note", "")
        if kind == "goto":
            return Action.goto(target)
        if kind == "fill":
            return Action.fill(target, value)
        if kind == "click":
            return Action.click(target)
        return Action.stop(note)


def llm_run_fn(*, model: str = "claude-opus-4-8", max_steps: int = 25):
    """
    A bench/run.py `RunFn` that drives an `LLMPolicy` against the app **mutated**
    for the cell's condition. A fresh policy and browser per case; the mutated app
    is what makes the matrix's seven columns mean something for a real agent.
    """
    from app.mutations import create_mutated_app

    async def run_fn(case, condition, trial) -> Transcript:
        app = create_mutated_app(case.db_path, mutations=condition.mutations)
        browser = TestClientBrowser(app)
        instruction = case.task.instruction("http://shop")
        return await run(instruction, browser, LLMPolicy(model=model), max_steps=max_steps)

    return run_fn


llm_run_fn.__test__ = False   # name starts with no "test", but keep parity/intent
