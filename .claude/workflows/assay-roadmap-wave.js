export const meta = {
  name: 'assay-roadmap-wave',
  description: 'Fan out independent Assay roadmap increments as parallel workers, then verify by the real test suite (never the workers\' self-report).',
  phases: [
    { title: 'Build', detail: 'one worker per independent increment, disjoint new files' },
    { title: 'Verify', detail: 'full pytest + bench.selfcheck across everything that landed' },
  ],
}

// ---------------------------------------------------------------------------
// The codified version of the 2026-10-04 Orca wave. Today it was coordinated by
// hand (coordinator decides the split, human approves, workers run, coordinator
// verifies). This script freezes that pathway so it is deterministic and
// reviewable as code: same WORK list -> same plan, every run.
//
// THE ONE INVARIANT that makes parallelism safe here: every item owns DISJOINT
// NEW files and touches no existing module. That is what lets the workers share
// one working tree. If a future item must edit a shared file, give it
// `isolation: 'worktree'` instead (a knob below), or sequence it in its own run.
//
// "Expand later" = append an entry to WORK. Nothing else changes.
// Run with:  Workflow({ name: 'assay-roadmap-wave' })   (re-runs the builders)
//            or edit WORK first to point at the NEXT increment.
// ---------------------------------------------------------------------------

const WORK = [
  {
    label: 'B2-customer-config',
    files: ['bench/customer.py', 'tests/test_customer.py'],
    spec: `Assay B2 — customer config layer. Prove the harness is NOT welded to our shop:
let a customer declare their OWN tasks + SQL checks against their OWN schema, with their own
seed/reset. Create ONLY bench/customer.py + tests/test_customer.py; edit no existing file.
Reuse bench.model (Check/CheckResult/Grade) READ-ONLY; import nothing from app/ so generality
holds at import time. Ship ONE worked example in a different domain than the shop (e.g. SaaS
billing: accounts + subscriptions). Enforce the CLAUDE.md calibration discipline: oracle passes
all checks, noop fails a named check, and the most plausible wrong agent fails a SPECIFIC named
side-effect check (baseline-keyed). Run only: python -m pytest tests/test_customer.py -q.`,
  },
  {
    label: 'B3-confidence-bands',
    files: ['bench/confidence.py', 'tests/test_confidence.py'],
    spec: `Assay B3 — grader confidence bands. Turn v2's false-success point counts into Wilson
score intervals. Create ONLY bench/confidence.py + tests/test_confidence.py; edit no existing
file. Pure stdlib (math + statistics.NormalDist, no numpy/scipy). Provide wilson_interval,
a ConfidenceBand dataclass, band(successes,total,confidence=0.95), and false_success_band(score)
reading a bench.graders.GraderScore READ-ONLY. Correct at the extremes (0/n, n/n, total=0).
Document why Wilson beats normal-approx. Tests assert hand-computed bounds + monotonicity.
Run only: python -m pytest tests/test_confidence.py -q.`,
  },
  {
    label: 'B3-receipt-grader',
    files: ['bench/receipt_grader.py', 'tests/test_receipt_grader.py'],
    spec: `Assay B3 — receipt grader. A non-SQL grader that parses /orders/latest into a structured
receipt (items, quantities, ship address, status) and judges against the task's expected receipt
— stronger than the DOM keyword scrape, so it catches a wrong QUANTITY. Create ONLY
bench/receipt_grader.py + tests/test_receipt_grader.py; edit no existing file. Import Verdict +
the Grader protocol from bench.graders READ-ONLY. Per-task checks for the 3 purchase tasks
(mirror _DOM_CHECKS). Document the screen-grader blind spot (can't see a side-effect order vs
baseline — SQL's job). Tests: correct buy passes, wrong-item fails, socks x1 fails / x3 passes.
Run only: python -m pytest tests/test_receipt_grader.py -q.`,
  },
]

const BUILD_SCHEMA = {
  type: 'object',
  properties: {
    label: { type: 'string' },
    files_written: { type: 'array', items: { type: 'string' } },
    own_tests_pass: { type: 'boolean' },
    summary: { type: 'string' },
  },
  required: ['label', 'files_written', 'own_tests_pass', 'summary'],
  additionalProperties: false,
}

phase('Build')
// Disjoint files -> safe to share the working tree. Flip to isolation:'worktree'
// per item only if it must touch a shared file.
const built = await parallel(
  WORK.map((w) => () =>
    agent(
      `${w.spec}\n\nWhen done, return the structured result: which files you wrote, whether your own test file passes, and a one-line summary.`,
      { label: w.label, phase: 'Build', schema: BUILD_SCHEMA }
    )
  )
)

const landed = built.filter(Boolean)
log(`built ${landed.length}/${WORK.length}; self-reported passing: ${landed.filter((b) => b.own_tests_pass).length}`)

phase('Verify')
// The workers do NOT grade their own homework. This is the external gate: the real
// suite + calibration over everything that landed, which is the only thing we trust.
const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    pytest_passed: { type: 'integer' },
    pytest_failed: { type: 'integer' },
    calibrated: { type: 'boolean' },
    regressions: { type: 'array', items: { type: 'string' } },
    verdict: { type: 'string', enum: ['accept', 'reject'] },
  },
  required: ['pytest_passed', 'pytest_failed', 'calibrated', 'verdict'],
  additionalProperties: false,
}
const verdict = await agent(
  `Independently verify the Assay repo after the parallel build above. In the repo root:
  1. Run: source .venv/bin/activate && python -m pytest -q   — record passed/failed counts.
  2. Run: python -m bench.selfcheck   — record whether it prints "calibrated".
  3. Run: git status --short   — flag any MODIFIED existing file (workers should only ADD new files;
     a modified existing file other than a known prior change is a red flag / possible conflict).
  Return the structured verdict. 'accept' only if pytest has 0 failures AND selfcheck is calibrated
  AND no unexpected existing file was modified; otherwise 'reject' with the regressions listed.`,
  { label: 'verify-suite', phase: 'Verify', schema: VERIFY_SCHEMA }
)

return { built: landed, verify: verdict }
