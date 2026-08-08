# sw-checklist Paydown Policy

**Status:** active 2026-05-04. Discipline that converges the
`sw-checklist` failure baseline to zero by treating it as a
debt that every commit pays down.

## Status snapshot -- 2026-07-27 (tech-debt spike complete)

The 2026-07 spike (saga tech-debt-spike, steps 001-016) took the
counts from 18 failed / 323 warnings to **1 failed / 296
warnings**. Retired: mlpl-viz (16 modules), autograd backward
fn-counts, mlpl-serve (17), mlpl-repl (14), mlpl-web (10),
mlpl-web-eval (20), plus a propagate Function-LOC FAIL and the
four repo-metadata warnings. The remaining FAIL is mlpl-eval's
100-module crate count -- owned by the dedicated plan in
docs/eval-decomposition-saga.md.

Recurring lesson for future ratchet work: a crate split retires
its FAIL but the new at-cap crates add structural module-count
warnings, and the >4 fn-count warn bar makes extraction-in-place
self-cancelling -- plan one extra offset retirement per split
step, or expect a documented exception.

## The rule

> Each commit must hold or LOWER the failed-count from its
> parent. Growth is forbidden except through the named
> exception lane below.

`sw-checklist` is the project's standards-and-budgets check
(function LOC, module function count, file LOC, crate module
count, clippy allows, etc.). It runs as part of `/mw-cp` and
prints a summary like `105 passed, 138 failed, 389 warnings`.
That `138` is what this policy targets.

## How it works

Every commit either:

1. **Holds** the count (`failed_count == prior - 0`). Allowed
   only when the commit is purely additive in a way that does
   not introduce a measurable check (e.g. a docs-only commit
   that doesn't touch Rust source).

2. **Pays down** the count (`failed_count <= prior - 1`).
   Required for any commit that introduces new code. The
   commit message includes a single-line "sw-checklist:" trailer
   noting the new count and which FAIL was retired.

3. **Exception lane.** A commit that genuinely cannot retire
   anything (e.g. a critical bugfix on a branch where every
   other refactor is pre-empted) may grow the count by at most
   1. The commit message MUST include `sw-checklist: exception`
   on its own line plus a one-paragraph justification.
   Exceptions are reviewed (manually) and ideally repaid in
   the next commit.

## Trajectory

The repo started at 139 failed (post-Saga 23). Today's commit
moves it to 138. At one retirement per commit, that is
~138 commits to green. A refactor saga that extracts an
oversized module can retire 3-5 in one go and shorten the
schedule.

## Easy retirement candidates

When a commit needs to pay down but doesn't have a natural
target, these are quick wins:

- **`#[allow(clippy::too_many_arguments)]`** -- refactor the
  fn signature to take a struct of args. Removes one
  `Clippy Allows` FAIL per fix.
- **Function LOC over 50** -- extract a helper or two.
- **Module function count over 7** -- move helpers into a
  sibling module.
- **File LOC over 500** -- split a large file into a small
  module group.

The full list lives in `sw-checklist -v` output. Pick whichever
is closest to the commit's main work to keep the diff cohesive.

## Warning-paydown reality (measured 2026-08-07)

A blanket "halve the warnings" spike is NOT worth pursuing and
can make things WORSE. Measured state at 382 warnings / 2 FAILs
(both documented transients):

- Function LOC: 165, Module Function Count: 157, Crate Module
  Count: 55, File LOC: 5.
- **78 modules sit at exactly 7 functions** (the FAIL ceiling),
  50 at 6, 29 at 5. Extracting a LOC helper into any 7-function
  module turns a Function-LOC WARNING into a Module-Function-
  Count FAIL. (Confirmed live: extracting a helper in
  random_builtins.rs, already at 7 fns, created a FAIL.)

The thresholds (warn at >4 fns/module, >4 modules/crate) are
below the natural size of almost every real module and crate,
so the warning total is closer to a structural EQUILIBRIUM than
to debt. The genuine floor is the FAIL line (>50 LOC/fn, >7
fns/module, >7 modules/crate), and only the 2 documented
transients sit there.

Therefore:

- **Do NOT** do blanket warning-reduction refactors. Every
  extraction in a saturated module trades one budget for
  another and risks creating FAILs.
- **DO** keep the per-commit ratchet on NEW code (land features
  under the gates so the total does not climb), and retire a
  warning only when it is one edit from crossing into a FAIL.
- A real reduction requires ARCHITECTURE work: splitting a
  genuinely-overloaded crate into sibling crates (each with its
  own module/fn budget), as `mlpl-runtime-bits` was carved from
  the runtime. Treat that as a scoped design decision per crate,
  not warning-golf.

## At zero

When the count hits 0, flip `sw-checklist` from advisory to
gating in `/mw-cp`. From that point any new FAIL fails the
build and must be fixed before merge. The exception lane goes
away.

## See also

- `CLAUDE.md` -- project-wide agent instructions; references
  this policy.
- `docs/sw-checklist-patterns.md` -- catalog of refactor
  patterns (struct-return, struct-args, validate-then-work,
  orchestrator + helpers, etc.) for retiring specific FAIL
  kinds.
