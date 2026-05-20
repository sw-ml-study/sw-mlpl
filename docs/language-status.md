# Language status dashboard

**One-screen view of where MLPL's language fixes stand.** The
catalog of findings lives in `docs/language-audit.md`; the queue
lives in `docs/plan.md`; this doc says what is *happening right
now* and what has *shipped*.

Update this doc whenever a saga step lands that changes the
status of any finding. Saga 30's step 006 doubles as the
audit-closeout step; analogous steps in later sagas should do the
same for their findings.

Last refreshed: 2026-05-20 (saga 30 step 001 shipped).

## Active saga

| Slug              | Status   | Steps total | Done | Next step                                |
|-------------------|----------|-------------|------|------------------------------------------|
| `tier1-cleanup`   | active   | 6           | 1    | 002 concat-axis-n-backward (audit #18)   |

`agentrail status` is the live source of truth; this row is the
human-readable summary.

## Saga timeline (oldest first)

| Saga                  | Status     | Audit findings touched | Notes                                       |
|-----------------------|------------|------------------------|---------------------------------------------|
| `vit` (29)            | shipped    | -- (capability saga)   | Closed 2026-05-20. Archived under `.agentrail-archive/`. |
| `tier1-cleanup` (30)  | active     | #18, #19               | This saga.                                  |
| Scripting cluster     | proposed   | #22, #24, #26, #28 (+ #23/#25/#27/#29/#30) | The four critical findings ship as one saga; if/else, args(), to_number, print. |
| Dim reduction         | proposed   | -- (capability saga)   | `docs/milestone-dimensionality-reduction.md`. UMAP-led. |
| Chronological history | proposed   | -- (content saga)      | `docs/milestone-chronological-history.md`. 24 per-concept lessons. |

The "proposed" sagas have full milestone docs; the user has
confirmed the editorial stances. They are not yet initialized in
agentrail.

## Per-finding status (critical tier)

| #   | Short title                                  | Status      | Owning saga / step       | Shipped commit |
|-----|----------------------------------------------|-------------|--------------------------|----------------|
| #1  | Closures don't differentiate                 | proposed    | future autograd-lift saga | --             |
| #2  | `device("mlx")` param relocation             | proposed    | future                   | --             |
| #3  | Booleans encoded as `0.0` / `1.0` floats     | proposed    | future                   | --             |
| #10 | No `vmap` / batched transform                | proposed    | future                   | --             |
| #12 | No `gather` / no slice ranges                | proposed    | future                   | --             |
| #15 | Inline forward expression anti-pattern       | downstream of #1 | --                  | --             |
| #18 | `concat` axis restricted to `{0, 1}`         | **in flight** | saga 30 (step 001 forward shipped; step 002 backward up next) | step 001: `c133d57` |
| #19 | Multi-head attention has forward-only tape   | queued      | saga 30 step 004         | --             |
| #22 | No `if` / `else`                             | proposed    | scripting saga           | --             |
| #24 | No CLI argument capture in script mode       | proposed    | scripting saga           | --             |
| #26 | No string-to-number parsing                  | proposed    | scripting saga           | --             |
| #28 | No `print` / explicit script output          | proposed    | scripting saga           | --             |

## Per-finding status (nice-to-have)

| #   | Short title                                  | Status   | Owning saga       |
|-----|----------------------------------------------|----------|-------------------|
| #4  | Magic seed constants                         | proposed | future            |
| #5  | `:upload` stringly-typed error kinds         | proposed | future            |
| #6  | `concat` arity overload / list-variadic      | proposed | (could rider on saga 30) |
| #8  | Stringly-typed `svg()` viz type names        | proposed | future            |
| #9  | Inconsistent axis position across builtins   | proposed | future            |
| #11 | No `jit` / compilation boundary              | proposed | future            |
| #14 | No named-axis types                          | proposed | saga 19 (queued)  |
| #16 | Model-DSL doesn't cover `take` / `reshape`   | proposed | future            |
| #17 | Stringly-typed device names                  | proposed | future            |
| #23 | No `while` / `break` / `continue`            | proposed | scripting saga    |
| #25 | No `env()`                                   | proposed | scripting saga    |
| #27 | No stdin read                                | proposed | scripting saga    |
| #29 | No script exit code                          | proposed | scripting saga    |
| #30 | No script-mode example demo                  | proposed | scripting saga    |

## Per-finding status (cosmetic)

| #   | Short title                                  | Status   |
|-----|----------------------------------------------|----------|
| #7  | Inconsistent builtin naming convention       | proposed |
| #13 | No tacit / point-free programming            | deferred (per audit) |
| #20 | `BUILTINS` slice has implicit ordering       | proposed |
| #21 | sw-checklist budget shapes the code          | process, out of scope for audit |

## Shipped (most recent first)

- **2026-05-20** -- saga 30 step 001: audit #18 forward lifted in
  `c133d57`. `mlpl-array::concat` now accepts any `axis` in
  `[0, rank)`. The autograd backward still restricts axis < 2;
  step 002 closes that.
