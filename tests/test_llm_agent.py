"""
The LLM brain, tested without a network or an API key.

`LLMPolicy` talks to Claude through an injected client, so a fake client that
returns scripted `act` tool calls lets us prove the whole seam offline:

- each tool-call `kind` becomes the right `Action`,
- a turn with no tool call ends the run instead of crashing it, and
- a scripted agent drives a real task through the real browser and the SQL grader
  passes it — exactly the path a live model takes, minus the model.

What a *live* model decides is not testable here (that's the experiment); that the
harness around it is correct, is.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from bench.agent import (
    LLMPolicy,
    Observation,
    TestClientBrowser,
    _affordances,
    run,
)
from bench.env import prepare_case
from bench.grader import grade
from bench.tasks import EMPTY_CART


# ----------------------------------------------------------- a fake Claude client

class _Block:
    def __init__(self, type_, name=None, input=None, text=""):
        self.type = type_
        self.name = name
        self.input = input
        self.text = text


class _Message:
    def __init__(self, blocks):
        self.content = blocks


class _Messages:
    def __init__(self, replies):
        self._replies = list(replies)
        self._i = 0
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        return reply


class FakeClient:
    """Returns pre-scripted messages; records every request for assertions."""

    def __init__(self, replies):
        self.messages = _Messages(replies)


def _act(kind, target="", value="", note=""):
    return _Message([_Block("tool_use", name="act",
                            input={"kind": kind, "target": target,
                                   "value": value, "note": note})])


OBS = Observation(url="http://shop/", html="<html></html>", text="")


@pytest.mark.parametrize("kind,target,value", [
    ("goto", "/cart", ""),
    ("click", "text=Add to cart", ""),
    ("fill", "#qty", "3"),
    ("stop", "", ""),
])
def test_tool_call_becomes_the_right_action(kind, target, value):
    client = FakeClient([_act(kind, target, value, note="n")])
    policy = LLMPolicy(client=client)
    action = policy.decide("task", OBS, ())
    assert action.kind == kind
    if kind in ("goto", "click", "fill"):
        assert action.target == target
    if kind == "fill":
        assert action.value == value


def test_a_turn_with_no_tool_call_stops_rather_than_crashes():
    client = FakeClient([_Message([_Block("text", text="I can't do that.")])])
    action = LLMPolicy(client=client).decide("task", OBS, ())
    assert action.kind == "stop"


def test_it_forces_a_tool_call_and_sends_the_task():
    client = FakeClient([_act("stop")])
    LLMPolicy(client=client).decide("buy the blue shirt", OBS, ())
    sent = client.messages.calls[0]
    assert sent["tool_choice"] == {"type": "any"}          # it must act
    assert [t["name"] for t in sent["tools"]] == ["act"]
    assert "buy the blue shirt" in sent["messages"][0]["content"]


def test_affordances_lists_buttons_and_fields(shop):
    client, _db = shop
    client.post("/login", data={"email": "alice@example.com", "password": "password123"})
    html = client.get("/product/SHIRT-BLU").text
    rendered = _affordances(html)
    assert "/cart/add" in rendered            # the real form action
    assert "name=sku" in rendered or "#qty" in rendered
    assert "Add to cart" in rendered          # the clickable label


def test_scripted_llm_agent_empties_the_cart_end_to_end(seed, tmp_path):
    """A fake model that reads nothing, just plays the right moves — and the SQL grader passes it."""
    case = prepare_case(EMPTY_CART, tmp_path / "empty", seed)
    client = FakeClient([
        _act("goto", "/login"),
        _act("fill", "#email", "alice@example.com"),
        _act("fill", "#password", "password123"),
        _act("click", "text=Log in"),
        _act("goto", "/cart"),
        _act("click", "text=Empty cart"),
        _act("stop", note="cart is empty, bought nothing"),
    ])
    browser = TestClientBrowser(create_app(case.db_path))
    import asyncio
    transcript = asyncio.run(
        run(EMPTY_CART.instruction("http://shop"), browser, LLMPolicy(client=client))
    )
    g = grade(case)
    assert g.passed, [r.name for r in g.results if not r.passed]
    assert transcript.stopped_reason == "cart is empty, bought nothing"
    assert not transcript.mistakes
