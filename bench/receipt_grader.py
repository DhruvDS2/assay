"""
B3: a receipt grader — judge success from the order receipt, not a keyword scrape.

The DOM grader (bench/graders.py) reads the rendered pages shallowly: does the word
"blue oxford shirt" appear somewhere on the order page? That's the check a quick
DOM-scraping harness would write. A receipt/confirmation-email grader reads deeper:
it parses the order-detail page into a *structured* receipt — line items with their
quantities, the ship-to address, the status — and judges against the task's expected
receipt. Same input page (`state['/orders/latest']`), stronger reading of it.

Parsing a receipt instead of substring-matching is what catches a wrong *quantity*:
a page that says "Wool Socks 1" contains the word "wool socks" just as a correct one
does, so a keyword scrape passes it; a receipt grader sees qty=1 against an expected
3 and fails it.

Blind spot (shared by every screen-based grader — DOM and screenshot included):
it reads the *latest* order's receipt and cannot see it against a baseline. It can't
tell a freshly placed order from one that was already there, and it can't see a
*second*, extraneous order placed off to the side — "exactly one NEW order" is a
side-effect question only SQL can answer by diffing against the pre-run snapshot.
The receipt confirms the order it can see is right; SQL confirms it's the only one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from bench.graders import Grader, Verdict  # read-only: the protocol + result type
from bench.model import Case


# ------------------------------------------------------------------- the receipt

@dataclass(frozen=True)
class Receipt:
    """The order-detail page parsed into the fields a receipt actually carries."""

    order_id: int | None
    status: str
    ship_address: str
    items: dict[str, int]          # product name (lowercased) -> quantity


def parse_receipt(order_text: str) -> Receipt:
    """
    Turn the visible text of `/orders/latest` into a `Receipt`.

    The order page renders a header (Status / Placed / Ship to) and a line-item
    table with a `Product Quantity Price` head and a closing `Total:` row. We
    anchor on those labels, so leading chrome (nav, inlined CSS) is ignored.
    """
    t = order_text.lower()

    m = re.search(r"order #(\d+)", t)
    order_id = int(m.group(1)) if m else None

    m = re.search(r"status:\s*(\w+)", t)
    status = m.group(1) if m else ""

    # Ship-to runs from its label up to the start of the line-item table header.
    m = re.search(r"ship to:\s*(.*?)\s*product\s+quantity\s+price", t)
    ship_address = m.group(1).strip() if m else ""

    # Line items live between the table header and the Total row; each row is a
    # product name, then an integer quantity, then a $price. Non-greedy name so
    # multi-word names ("blue oxford shirt") stay whole and don't swallow the qty.
    items: dict[str, int] = {}
    body = re.search(r"product\s+quantity\s+price(.*?)total:", t, re.S)
    if body:
        for name, qty in re.findall(r"([a-z][a-z ]+?)\s+(\d+)\s+\$[\d.]+", body.group(1)):
            items[name.strip()] = items.get(name.strip(), 0) + int(qty)

    return Receipt(order_id=order_id, status=status, ship_address=ship_address, items=items)


# ------------------------------------------------------------ per-task receipt checks
# One check per purchase task, keyed by task.id — mirrors `_DOM_CHECKS`. Each takes a
# parsed Receipt and returns (passed, why). We build them from a tiny spec so each
# reads like what the receipt *should* say: these exact items at these quantities,
# shipped to this address, status placed.


def _receipt_matches(want_items: dict[str, int], address_fragment: str) -> Callable[[Receipt], tuple[bool, str]]:
    def check(r: Receipt) -> tuple[bool, str]:
        if r.status != "placed":
            return False, f"status is {r.status!r} (want 'placed')"
        if r.items != want_items:
            return False, f"receipt items {r.items}, want {want_items}"
        if address_fragment.lower() not in r.ship_address:
            return False, f"ship-to {r.ship_address!r} missing {address_fragment!r}"
        return True, f"receipt shows {want_items} shipped to {address_fragment}"
    return check


_RECEIPT_CHECKS: dict[str, Callable[[Receipt], tuple[bool, str]]] = {
    "buy_blue_shirt": _receipt_matches({"blue oxford shirt": 1}, "12 Elm St"),
    "buy_socks_x3": _receipt_matches({"wool socks": 3}, "400 Main St"),
    "buy_tote_and_mug": _receipt_matches({"canvas tote": 1, "enamel mug": 1}, "9 Oak Ave"),
}


class ReceiptGrader:
    """Judge a purchase from the parsed order receipt — items, quantities, address."""

    name = "receipt"

    def judge(self, case: Case, state: dict[str, str]) -> Verdict:
        check = _RECEIPT_CHECKS.get(case.task.id)
        if check is None:
            # Out of domain (not a purchase): abstain so the v2 comparison doesn't
            # score this grader on tasks it was never meant to judge.
            return Verdict(self.name, False, "no receipt check for this task", abstained=True)
        receipt = parse_receipt(state.get("/orders/latest", ""))
        ok, why = check(receipt)
        return Verdict(self.name, ok, why)


_: Grader = ReceiptGrader()   # structural: ReceiptGrader satisfies the Grader protocol
