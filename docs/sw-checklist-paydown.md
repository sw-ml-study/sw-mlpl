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
