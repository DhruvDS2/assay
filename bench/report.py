"""
B1: the report — a static, self-contained HTML artifact of a matrix run.

This is the sales-proof. A `RunSummary` is honest but terse; this turns it into a
page a human reads top-to-bottom: the headline numbers, the matrix with the
**worst cells first** (a green wall hides nothing; the broken cells are what a
buyer needs to see), and for every failure the exact ground-truth checks that
caught it — the trace that only owning the database can produce ("2 orders (want
1)", not "the screenshot looked wrong").

Two design choices keep it a *trust* artifact, not a demo:

  * **Worst-first.** Cells sort by pass rate ascending; skipped (infra never let
    the agent try) sort above failures, because an un-run cell is a worse thing to
    hide than a failed one. You land on the problems.
  * **The trace is SQL, not vibes.** Each failed trial lists its failed check names
    and their human-readable detail from `bench/grader.py` — the database's own
    account of what went wrong. Optionally the v2 grader-comparison table rides
    along, so the report also shows how often a *cheaper* judge would have lied.

Self-contained: one HTML string, inline CSS, no JS, no external assets — it opens
anywhere and can be emailed as-is. (rrweb step-replay is the next increment; it
needs per-run recordings the keyless harness doesn't capture yet.)
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Optional

from bench.run import Cell, RunSummary

_CSS = """\
:root { --ok:#1a7f37; --warn:#9a6700; --bad:#cf222e; --skip:#6e7781; --line:#d0d7de; }
* { box-sizing: border-box; }
body { font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; color:#1f2328;
       margin:0; padding:2rem; max-width:980px; margin-inline:auto; }
h1 { margin:0 0 .25rem; } h2 { margin:2rem 0 .5rem; border-bottom:1px solid var(--line);
     padding-bottom:.25rem; } .sub { color:var(--skip); margin-top:0; }
.cards { display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }
.card { border:1px solid var(--line); border-radius:8px; padding:.75rem 1rem; min-width:9rem; }
.card .n { font-size:1.6rem; font-weight:700; } .card .l { color:var(--skip); font-size:.85rem; }
table { border-collapse:collapse; width:100%; margin:.5rem 0; }
th,td { text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--line); }
th { font-size:.8rem; text-transform:uppercase; letter-spacing:.03em; color:var(--skip); }
.badge { display:inline-block; padding:.05rem .5rem; border-radius:999px; font-size:.8rem;
         font-weight:600; color:#fff; }
.ok{background:var(--ok)} .warn{background:var(--warn)} .bad{background:var(--bad)} .skip{background:var(--skip)}
tr.row-bad td:first-child { border-left:3px solid var(--bad); }
tr.row-warn td:first-child { border-left:3px solid var(--warn); }
tr.row-skip td:first-child { border-left:3px solid var(--skip); }
tr.row-ok td:first-child { border-left:3px solid var(--ok); }
details { margin:.5rem 0; } summary { cursor:pointer; font-weight:600; }
.trace { margin:.4rem 0 .4rem 1rem; }
.trace li { color:var(--bad); } .trace .detail { color:var(--skip); font-family:ui-monospace,monospace; }
.mono { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }
footer { margin-top:2.5rem; color:var(--skip); font-size:.85rem; }
"""


def _pct(x: Optional[float]) -> str:
    return f"{x:.0%}" if x is not None else "–"


def _cell_class(c: Cell) -> str:
    if c.attempted == 0:
        return "skip"
    if c.pass_hat_k:
        return "ok"
    if c.passed == 0:
        return "bad"
    return "warn"


def _worst_first(cells) -> list[Cell]:
    # Skipped (never attempted) first, then lowest pass-rate up. Reliable cells sink.
    def key(c: Cell):
        rate = c.pass_rate if c.pass_rate is not None else -1.0
        return (rate, c.task_id, c.condition)
    return sorted(cells, key=key)


def _matrix_rows(cells) -> str:
    rows = []
    for c in _worst_first(cells):
        kind = _cell_class(c)
        label = {"ok": "pass^k", "warn": "flaky", "bad": "fail", "skip": "skipped"}[kind]
        passes = f"{c.passed}/{c.attempted}" if c.attempted else "–"
        rows.append(
            f'<tr class="row-{kind}">'
            f"<td class=mono>{escape(c.task_id)}</td>"
            f"<td class=mono>{escape(c.condition)}</td>"
            f"<td>{passes}</td><td>{_pct(c.pass_rate)}</td>"
            f"<td>{c.skipped}</td>"
            f'<td><span class="badge {kind}">{label}</span></td>'
            "</tr>"
        )
    return "\n".join(rows)


def _failure_traces(cells) -> str:
    """For every cell that isn't clean pass^k, the ground-truth account per trial."""
    blocks = []
    for c in _worst_first(cells):
        if c.pass_hat_k:
            continue
        items = []
        for g in c.grades:
            if g.status == "skipped":
                items.append(f"<li>trial {g.trial}: <b>skipped</b> — infra never let the agent run"
                             f"{(' ('+escape(str(g.error))+')') if g.error else ''}</li>")
                continue
            if g.status == "passed":
                items.append(f"<li>trial {g.trial}: passed</li>")
                continue
            failed = [r for r in (g.grade.results if g.grade else ()) if not r.passed]
            detail = "".join(
                f'<li>{escape(r.name)} — <span class=detail>{escape(r.detail)}</span></li>'
                for r in failed
            ) or "<li>(no check detail recorded)</li>"
            items.append(
                f"<li>trial {g.trial}: <b>failed</b>"
                f'<ul class="trace">{detail}</ul></li>'
            )
        blocks.append(
            f"<details><summary>{escape(c.task_id)} / {escape(c.condition)} "
            f"— {c.passed}/{c.attempted or 0} passed</summary>"
            f'<ul class="trace">{"".join(items)}</ul></details>'
        )
    return "\n".join(blocks) if blocks else "<p>Every cell passed every trial. Nothing to explain.</p>"


def _grader_section(grader_scores) -> str:
    if not grader_scores:
        return ""
    from bench.confidence import false_success_band

    rows = []
    for name, s in grader_scores.items():
        kind = "bad" if s.false_success else "ok"
        b = false_success_band(s)
        rows.append(
            f"<tr><td class=mono>{escape(name)}</td>"
            f"<td>{s.agree}/{s.total}</td>"
            f'<td><span class="badge {kind}">{s.false_success}/{s.total} ({s.false_success_rate:.0%})</span></td>'
            f"<td class=mono>[{b.lower:.0%}&ndash;{b.upper:.0%}]</td>"
            f"<td>{s.false_failure}/{s.total}</td></tr>"
        )
    return (
        "<h2>Cheaper judges, scored against SQL truth</h2>"
        "<p class=sub>How often a DOM scrape or screenshot judge would have certified "
        "the agent as done when the database says it wasn't. False-success is the "
        "dangerous one: it silently inflates agent scores everywhere SQL isn't available. "
        "The 95% CI is a Wilson band &mdash; a 0/70 is &ldquo;0, give or take&rdquo;, not proven perfect.</p>"
        "<table><thead><tr><th>grader</th><th>agree</th><th>false-success</th>"
        "<th>95% CI</th><th>false-failure</th></tr></thead><tbody>"
        + "\n".join(rows) + "</tbody></table>"
    )


def render_report(
    summary: RunSummary,
    *,
    title: str = "Assay — robustness report",
    subtitle: str = "",
    grader_scores=None,
) -> str:
    s = summary
    cards = [
        ("passed / planned", f"{s.passed}/{s.planned}", _pct(s.passed_over_planned)),
        ("passed / attempted", f"{s.passed}/{s.attempted}", _pct(s.passed_over_attempted)),
        ("reliable cells", f"{len(s.reliable_cells)}/{len(s.cells)}", "pass^k"),
        ("skipped (infra)", str(s.skipped), "not counted as fail"),
    ]
    card_html = "".join(
        f'<div class=card><div class=n>{escape(v)}</div>'
        f'<div class=l>{escape(l)}</div><div class=l>{escape(sub)}</div></div>'
        for l, v, sub in cards
    )
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>{escape(title)}</title><style>{_CSS}</style></head>
<body>
<h1>{escape(title)}</h1>
<p class=sub>{escape(subtitle) if subtitle else 'Grade by database, not by screenshot. Worst cells first.'}</p>
<div class=cards>{card_html}</div>

<h2>Matrix — worst cells first</h2>
<table><thead><tr><th>task</th><th>condition</th><th>pass</th><th>rate</th>
<th>skip</th><th>verdict</th></tr></thead>
<tbody>
{_matrix_rows(s.cells)}
</tbody></table>

<h2>Why the failures failed — the ground-truth trace</h2>
<p class=sub>Each failed trial, with the exact named checks the database used to catch it.</p>
{_failure_traces(s.cells)}

{_grader_section(grader_scores)}

<footer>Generated by <span class=mono>bench/report.py</span>. SQL is the truth;
every verdict above is a query against the shop's own SQLite, not a guess about a page.</footer>
</body></html>"""


def write_report(summary: RunSummary, path: str | Path, **kw) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(summary, **kw), encoding="utf-8")
    return path


# ----------------------------------------------------------------------- CLI
# Generate a report from a real run: the hardened agent across the fault axis
# (so the matrix has honest passes AND failures), with the v2 grader-comparison
# table riding along. Writes a single self-contained file under runs/ (gitignored).


def main() -> int:
    import asyncio
    import tempfile

    from bench.gradecompare import DEFAULT_RUNS, default_graders, run_comparison
    from bench.graders import false_success_rate
    from bench.hardened import hardened_recovery_run_fn
    from bench.run import fault_conditions, run_matrix, seven_conditions
    from bench.tasks import BUY_BLUE_SHIRT, BUY_SOCKS_X3, BUY_TOTE_AND_MUG

    purchase = (BUY_BLUE_SHIRT, BUY_SOCKS_X3, BUY_TOTE_AND_MUG)

    with tempfile.TemporaryDirectory() as tmp:
        summary, _ = asyncio.run(
            run_matrix(purchase, fault_conditions(), trials=1,
                       root=f"{tmp}/m", run_fn=hardened_recovery_run_fn())
        )
        graders, _live = default_graders()
        cells = run_comparison(DEFAULT_RUNS, seven_conditions(), graders, root=f"{tmp}/g")
        scores = false_success_rate([list(c.verdicts) for c in cells])

    out = write_report(
        summary,
        "runs/report.html",
        title="Assay — robustness report",
        subtitle="Hardened agent across the fault axis · cheap graders scored against SQL truth.",
        grader_scores=scores,
    )
    print(f"wrote {out}  ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
