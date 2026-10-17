export const meta = {
  name: 'assay-roadmap-wave',
  description: 'Fan out independent Assay roadmap increments as parallel workers, then verify by the real test suite (never the workers\' self-report).',
  phases: [
    { title: 'Build', detail: 'one worker per independent increment, disjoint new files' },
    { title: 'Verify', detail: 'full pytest + bench.selfcheck across everything that landed' },
  ],
}

// ---------------------------------------------------------------------------
// The codified version of the hand-coordinated Orca wave: coordinator decides
// the split, human approves, workers run on DISJOINT NEW files, coordinator
// verifies by the real suite. This script freezes that pathway deterministically.
//
// THE ONE INVARIANT that makes parallelism safe here: every item owns DISJOINT
// NEW files and touches no existing module. That is what lets the workers share
// one working tree. If a future item must edit a shared file, give it
// `isolation: 'worktree'` instead, or sequence it in its own run.
//
// HOW TO DRIVE IT (args-first — the increments are DATA, passed in):
//   Workflow({ name: 'assay-roadmap-wave', args: [
//     { label: '...', files: ['bench/x.py','tests/test_x.py'],
//       expectModified: ['bench/__init__.py'],   // optional: known pre-existing edits, not regressions
//       spec: `...worker instructions...` },
//     ...
//   ] })
// A bare `Workflow({ name: 'assay-roadmap-wave' })` falls back to DEFAULT_WORK
// below; DEFAULT_WORK is intentionally EMPTY so a no-arg run is a loud no-op,
// never a silent re-build of already-landed increments (the 2026-10-16 gotcha).
// Append completed waves to DONE_LOG for the audit trail; don't re-queue them.
// ---------------------------------------------------------------------------

// Landed waves (audit trail — do NOT put these back in WORK; they already exist):
//   2026-10-04  B2-customer-config, B3-confidence-bands, B3-receipt-grader
//   2026-10-16  A2-measured (bench/a2_measure.py), A3-site-memory (bench/sitegraph.py),
//               B3-review-queue (bench/review_queue.py)
const DEFAULT_WORK = []

// `args` is the queue of increments to build this run. Each: {label, files[], spec, expectModified?[]}.
const WORK = Array.isArray(args) && args.length ? args : DEFAULT_WORK

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
if (!WORK.length) {
  log('no increments queued: pass args=[{label, files, spec}, ...] to build a wave. DEFAULT_WORK is empty by design.')
  return { built: [], verify: null, note: 'no-op: no increments passed' }
}

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
const expectedNewFiles = WORK.flatMap((w) => w.files || [])
const expectModified = WORK.flatMap((w) => w.expectModified || [])
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
  3. Confirm the NEW files for this wave actually landed (a worker that reported "already exists"
     without writing is a FAIL): ${JSON.stringify(expectedNewFiles)}.
  4. Run: git status --short   — workers should only ADD new files. These existing files are
     KNOWN-EXPECTED edits for this wave, NOT regressions: ${JSON.stringify(expectModified)}.
     Any OTHER modified tracked file is a red flag / possible conflict.
  Return the structured verdict. 'accept' only if pytest has 0 failures AND selfcheck is calibrated
  AND every expected new file exists AND no unexpected existing file was modified; else 'reject'
  with the regressions listed.`,
  { label: 'verify-suite', phase: 'Verify', schema: VERIFY_SCHEMA }
)

return { built: landed, verify: verdict }
