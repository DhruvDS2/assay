"""
Site knowledge-graph memory — the A3 (v5) experiment seam.

The benchmarked agent, over its runs, builds a graph of the *site it is driving*:

    pages  ->  elements  ->  actions  ->  outcomes

A **page** is a URL path (`/cart`, `/checkout`). On each page live the
**elements** an agent can act on — the links, buttons, and fields the offline
browser already parses out of the plain HTML. Taking an **action** on an element
(click this button, follow this link, go to this path) produces an **outcome**:
the page you land on, and whether the DB-visible world changed ("Place order on
/checkout creates an order"). The graph remembers those outcomes so that next
time, before acting, the agent can *consult* what it learned instead of
re-discovering the site from scratch.

Three operations, matching the roadmap's "build / consult / update":

    build    — observe a page and register its elements as graph nodes
    consult  — given an intent on a page, recall an action that worked before
    update   — after acting, record (page --action--> outcome page)

`MemoryAugmentedPolicy` wires those into the `bench.agent` loop: it wraps *any*
`Policy`, consults the graph before letting the inner policy decide, lets the
page it observed update the graph, and records the transition after the fact.

The whole point of the experiment is the **stale-graph hazard**: a graph built
on the clean site is a liability once the site mutates. If `rename_controls`
turns "Add to cart" into "Add to bag", the graph still points the agent at a
control labelled "Add to cart" — a control the mutated page no longer has. Memory
that helped on the clean site now *misleads*. `consult` reports whether the
recalled element still matches what is actually on the page so an experiment can
measure exactly that: helps clean, hurts stale. Measured against the A1
no-memory baseline (the bare inner policy with no wrapper).

This module is pure and offline: it reads the HTML the browser already produced
(via `bench.agent._PageModel`) and never touches the network. It imports only
from `bench.agent`, and changes no file outside itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
from urllib.parse import urlsplit

from bench.agent import (
    Action,
    Observation,
    Policy,
    Step,
    _PageModel,
)


# --------------------------------------------------------------------- page keys


def page_key(url: str) -> str:
    """
    Normalise a URL to the graph's notion of a *page*: its path, without query
    or host. `http://shop/cart?x=1` and `/cart` are the same page node, so a
    graph built behind one base URL still matches a run behind another.
    """
    if not url:
        return ""
    parts = urlsplit(url)
    if parts.path:
        return parts.path
    # A bare path with no scheme/host (what goto targets look like).
    return url.split("?", 1)[0] or "/"


# ----------------------------------------------------------------- graph nodes


@dataclass(frozen=True)
class Element:
    """
    One actionable thing on a page — the agent-facing surface, keyed by the
    *label* an agent reads (a link's text, a button's text, a field's selector).

    `label` is what the agent keys off and what a mutation can rename out from
    under it; `kind` says how to act on it; `selector`/`target` is the action's
    concrete target in the browser's vocabulary.
    """

    kind: str          # "link" | "button" | "field"
    label: str         # visible text (link/button) or field selector token
    target: str        # click target (text=... or path) / fill selector
    href: str = ""     # for links: the path they go to (an edge hint)


@dataclass
class PageNode:
    """A page in the graph: its elements, plus the outcomes of acting on it."""

    key: str
    elements: list[Element] = field(default_factory=list)
    # label -> outcome page key last seen after acting on that element.
    transitions: dict[str, str] = field(default_factory=dict)
    # labels whose action was observed to create a new order (a DB outcome).
    creates_order: set[str] = field(default_factory=set)
    visits: int = 0

    def element_for_label(self, label: str) -> Optional[Element]:
        want = label.strip().lower()
        for el in self.elements:
            if el.label.strip().lower() == want:
                return el
        # Substring match: "Add to cart" recalled, page shows "Add to cart now".
        for el in self.elements:
            if want and want in el.label.strip().lower():
                return el
        return None


# ------------------------------------------------------------------- recall type


@dataclass(frozen=True)
class Recall:
    """
    The answer `consult` gives: an action the graph recommends for an intent on
    a page, plus whether the element it points at *still matches* the page as it
    is right now. `stale=True` is the hazard signal — the graph is steering the
    agent at a control the (mutated) page no longer has.
    """

    action: Optional[Action]
    element: Optional[Element]
    stale: bool
    reason: str

    @property
    def hit(self) -> bool:
        return self.action is not None


# ------------------------------------------------------------- intent vocabulary
# An *intent* is what the agent wants to accomplish, independent of the exact
# wording on the page. The graph maps an intent to the element it learned serves
# it. Keeping this a small, explicit table (not an LLM) keeps the experiment
# offline and deterministic — the hazard is about labels drifting, not reasoning.


# intent -> the labels (lowercase) that have historically served it, best first.
_INTENT_LABELS: dict[str, tuple[str, ...]] = {
    "add_to_cart": ("add to cart",),
    "place_order": ("place order",),
    "empty_cart": ("empty cart",),
    "cancel_order": ("cancel order",),
    "checkout": ("proceed to checkout", "checkout"),
    "update_cart": ("update",),
}


def intent_for(label: str) -> Optional[str]:
    """Reverse of `_INTENT_LABELS`: which intent a learned label serves, if any."""
    want = label.strip().lower()
    for intent, labels in _INTENT_LABELS.items():
        if want in labels:
            return intent
    return None


# ----------------------------------------------------------------- the graph


class SiteGraph:
    """
    The agent's memory of a site: pages -> elements -> actions -> outcomes.

    Empty at birth. `build` fills it from pages the agent observes; `consult`
    recalls a learned action for an intent; `update` records what an action
    actually did. All three are pure and offline.
    """

    def __init__(self) -> None:
        self.pages: dict[str, PageNode] = {}

    # --- build ---------------------------------------------------------------

    def build(self, observation: Observation) -> PageNode:
        """
        Register (or refresh) the page the agent is looking at: parse its HTML
        into elements and store them as graph nodes. Idempotent — re-observing a
        page replaces its element list with what's on the page *now* (so a graph
        kept live across a run tracks the real site; a graph frozen after a clean
        run is what goes stale).
        """
        key = page_key(observation.url)
        node = self.pages.get(key)
        if node is None:
            node = PageNode(key=key)
            self.pages[key] = node
        node.elements = _elements_of(observation.html)
        node.visits += 1
        return node

    def has_page(self, url: str) -> bool:
        return page_key(url) in self.pages

    def page(self, url: str) -> Optional[PageNode]:
        return self.pages.get(page_key(url))

    # --- consult -------------------------------------------------------------

    def consult(self, intent: str, observation: Observation) -> Recall:
        """
        Recall an action the graph learned for `intent` on the current page, and
        report whether the element it points at is still on the page.

        A miss (`action is None`) means the graph has nothing to offer — the
        agent falls back to its own policy (this is also the A1 baseline's always
        path, since it never has a graph). A hit with `stale=True` is the hazard:
        the graph recalled a label that no longer exists on the (mutated) page.
        """
        key = page_key(observation.url)
        node = self.pages.get(key)
        if node is None:
            return Recall(None, None, stale=False,
                          reason=f"no memory of page {key!r}")

        remembered = self._remembered_element(node, intent)
        if remembered is None:
            return Recall(None, None, stale=False,
                          reason=f"no learned action for intent {intent!r} on {key!r}")

        # Does the control the graph remembers still exist on the page *now*?
        live = _elements_of(observation.html)
        match = _find_by_label(live, remembered.label)
        if match is None:
            # The hazard: memory points at a control the current page lacks.
            return Recall(
                self._action_for(remembered), remembered, stale=True,
                reason=(f"stale: recalled {remembered.label!r} for {intent!r} "
                        f"but it is not on {key!r} now"),
            )
        return Recall(
            self._action_for(match), match, stale=False,
            reason=f"recalled {match.label!r} for {intent!r} on {key!r}",
        )

    # --- update --------------------------------------------------------------

    def update(
        self,
        before: Observation,
        action: Action,
        after: Observation,
        *,
        order_created: bool = False,
    ) -> None:
        """
        Record the outcome of `action` taken on the `before` page: the page it
        led to, and (if known) that it created an order — the "Place order on
        /checkout creates an order" edge. Keeps the graph's transitions honest to
        what was actually observed, which is what lets a *live* graph adapt and a
        *frozen* one fall behind.
        """
        if action.kind not in ("click", "goto", "fill"):
            return
        from_key = page_key(before.url)
        node = self.pages.get(from_key) or self.build(before)

        label = _label_of_action(action)
        to_key = page_key(after.url)
        if label:
            node.transitions[label] = to_key
            if order_created:
                node.creates_order.add(label)

    # --- internals -----------------------------------------------------------

    def _remembered_element(self, node: PageNode, intent: str) -> Optional[Element]:
        """The element on `node` that historically served `intent`, if learned."""
        labels = _INTENT_LABELS.get(intent, ())
        for want in labels:
            el = node.element_for_label(want)
            if el is not None:
                return el
        return None

    @staticmethod
    def _action_for(el: Element) -> Action:
        if el.kind == "field":
            return Action.fill(el.target, "")
        return Action.click(el.target)


# --------------------------------------------------------------- HTML -> elements


def _elements_of(html: str) -> list[Element]:
    """Parse a page's actionable surface into graph `Element`s (links, buttons, fields)."""
    page = _PageModel.parse(html)
    out: list[Element] = []
    for lk in page.links:
        if lk.text:
            out.append(Element(kind="link", label=lk.text,
                               target=f"text={lk.text}", href=lk.href))
    for form in page.forms:
        for b in form.buttons:
            if b:
                out.append(Element(kind="button", label=b, target=f"text={b}"))
        for name in form.fields:
            ids = [fid for fid, nm in form.ids.items() if nm == name]
            sel = f"#{ids[0]}" if ids else f"name={name}"
            out.append(Element(kind="field", label=sel, target=sel))
    return out


def _find_by_label(elements: Sequence[Element], label: str) -> Optional[Element]:
    want = label.strip().lower()
    for el in elements:
        if el.label.strip().lower() == want:
            return el
    return None


def _label_of_action(action: Action) -> str:
    """The graph key for an action: its visible-label if a click, else its target."""
    t = action.target
    if action.kind == "click" and t.startswith("text="):
        return t[len("text="):].strip().strip("\"'")
    return t


# ------------------------------------------------------- memory-augmented policy


class MemoryAugmentedPolicy:
    """
    Wrap any `bench.agent` `Policy` with a `SiteGraph` it consults before acting
    and updates after. The inner policy stays in charge of *sequencing* (what to
    do next); memory contributes the learned *specifics* (which concrete control
    serves that intent) — the division a real agent would keep too.

    Per `decide`:
      1. `build` the current page into the graph (observe before act).
      2. Ask the inner policy for its next action — this keeps the plan's order.
      3. If that action is a click whose intent the graph knows, `consult` the
         graph and *remap the label* to the control memory learned:
         - a **fresh** recall (the control is on the page) remaps to it — memory
           helping; on a clean site this is a no-op or a harmless re-label.
         - a **stale** recall (the control is gone, e.g. renamed by a mutation)
           is the hazard: a credulous policy (`trust_stale=True`) follows memory
           onto the vanished control and misfires; otherwise memory steps aside
           and the inner action stands (so staleness never *helps* the wrapper
           beat the A1 baseline).
      4. Non-click actions, or clicks with no learned intent, pass through
         untouched — so a correct inner script is never derailed.

    Then `observe_result` feeds the post-action page back so `update` can record
    the transition. The loop in `bench.agent.run` doesn't call `observe_result`
    itself (it's policy-agnostic), so a `MemoryAugmentedPolicy` driven directly
    in a test or an A3 harness calls it between steps; when it isn't called, the
    policy still works — it just learns only elements, not transitions.
    """

    def __init__(
        self,
        inner: Policy,
        graph: Optional[SiteGraph] = None,
        *,
        learn: bool = True,
        trust_stale: bool = False,
    ) -> None:
        self.inner = inner
        self.graph = graph if graph is not None else SiteGraph()
        self.learn = learn
        self.trust_stale = trust_stale
        # Diagnostics an experiment reads to tell helps-clean from hurts-stale.
        self.consulted = 0
        self.memory_hits = 0
        self.stale_hits = 0
        self.fallbacks = 0
        self.last_recall: Optional[Recall] = None
        # Remember the last decision so observe_result can close the edge.
        self._pending: Optional[tuple[Observation, Action]] = None

    def decide(
        self, instruction: str, observation: Observation, history: Sequence[Step]
    ) -> Action:
        if self.learn:
            self.graph.build(observation)

        # The inner policy owns sequencing; memory only refines the chosen control.
        proposed = self.inner.decide(instruction, observation, history)
        self.last_recall = None

        intent = self._intent_of(proposed, instruction, observation, history)
        if intent is None:
            self.fallbacks += 1
            return self._remember(observation, proposed)

        recall = self.graph.consult(intent, observation)
        self.last_recall = recall
        if not recall.hit:
            self.fallbacks += 1
            return self._remember(observation, proposed)

        self.consulted += 1
        if recall.stale:
            self.stale_hits += 1
            if self.trust_stale:
                # Credulous: follow stale memory onto the vanished control.
                return self._remember(observation, recall.action)
            # Skeptical: drop the stale recall, keep the inner action.
            self.fallbacks += 1
            return self._remember(observation, proposed)

        # Fresh recall: adopt the control memory learned serves this intent.
        self.memory_hits += 1
        return self._remember(observation, recall.action)

    def _remember(self, observation: Observation, action: Action) -> Action:
        self._pending = (observation, action)
        return action

    def observe_result(
        self, after: Observation, *, order_created: bool = False
    ) -> None:
        """Feed the page reached after the last action so the graph learns the edge."""
        if self._pending is None or not self.learn:
            return
        before, action = self._pending
        self.graph.update(before, action, after, order_created=order_created)
        self._pending = None

    # --- intent inference ----------------------------------------------------

    def _intent_of(
        self,
        proposed: Action,
        instruction: str,
        observation: Observation,
        history: Sequence[Step],
    ) -> Optional[str]:
        """
        Which learned intent (if any) the inner policy's *proposed click* serves.

        Memory only speaks for clicks. Two ways to read the intent, in order:
          1. the proposed label itself, if the intent table knows it (clean site,
             the agent clicks "Add to cart" verbatim);
          2. else the *page context* — on `/product/*` a click is add-to-cart, on
             `/checkout` it's place-order, etc. This is the crucial case under a
             mutation: the inner correctly clicks the renamed "Add to bag", whose
             label the table does NOT know, yet the agent's *intent* is still
             add-to-cart — exactly where a clean-built graph gets to mislead it.

        Deliberately simple and offline, so the experiment isolates the *label
        drift* hazard rather than reasoning quality. Non-clicks return None and
        pass through untouched.
        """
        if proposed.kind != "click":
            return None
        by_label = intent_for(_label_of_action(proposed))
        if by_label is not None:
            return by_label
        return self._page_intent(observation, instruction)

    @staticmethod
    def _page_intent(observation: Observation, instruction: str) -> Optional[str]:
        """The intent a click on the current page most likely serves, by URL + task."""
        key = page_key(observation.url)
        goal = instruction.lower()
        if key.startswith("/product"):
            return "add_to_cart"
        if key == "/checkout":
            return "place_order"
        if key == "/cart":
            if "empty" in goal or "remove everything" in goal:
                return "empty_cart"
            return None          # cart has many controls; don't guess checkout
        if key.startswith("/orders") and "cancel" in goal:
            return "cancel_order"
        return None


MemoryAugmentedPolicy.__test__ = False  # not a pytest test
