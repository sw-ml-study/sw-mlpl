# Language status dashboard

**One-screen view of where MLPL's language fixes stand.** The
catalog of findings lives in `docs/language-audit.md`; the queue
lives in `docs/plan.md`; this doc says what is *happening right
now* and what has *shipped*.

Update this doc whenever a saga step lands that changes the
status of any finding. Saga 30's step 006 doubles as the
audit-closeout step; analogous steps in later sagas should do the
same for their findings.

Last refreshed: 2026-05-21 (saga 31 step 002 shipped).

## Active saga

| Slug                | Status   | Steps total | Done | Next step                                |
|---------------------|----------|-------------|------|------------------------------------------|
| `scripting-cluster` | active   | 8           | 2    | 003 args-builtin-and-cli-passthrough (audit #24) |

`agentrail status` is the live source of truth; this row is the
human-readable summary.

## Saga timeline (oldest first)

| Saga                  | Status     | Audit findings touched | Notes                                       |
|-----------------------|------------|------------------------|---------------------------------------------|
| `vit` (29)            | shipped    | -- (capability saga)   | Closed 2026-05-20. Archived under `.agentrail-archive/`. |
| `tier1-cleanup` (30)  | shipped    | #18, #19               | Closed 2026-05-20. Six steps; both findings retired. |
| `scripting-cluster` (31) | active  | #22, #23, #24, #25, #26, #27, #28, #29, #30 | This saga. 8 steps; turns MLPL into a real scripting language. |
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
| #18 | `concat` axis restricted to `{0, 1}`         | **shipped** | saga 30 steps 001 (forward) + 002 (backward) | 001: `c133d57`, 002: `4e27f9c` |
| #19 | Multi-head attention has forward-only tape   | **shipped (stale audit)** | originally saga 29 step 013; verified in saga 30 step 004 | `66d63c9` |
| #22 | No `if` / `else`                             | proposed    | scripting saga           | --             |
| #24 | No CLI argument capture in script mode       | proposed    | scripting saga           | --             |
| #26 | No string-to-number parsing                  | **shipped** | saga 31 step 002         | `87f4a2b`      |
| #28 | No `print` / explicit script output          | **shipped** | saga 31 step 001         | `4f7f1f2`      |

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
| #25 | No `env()`                                   | **shipped** (saga 31 step 002, `87f4a2b`) | scripting saga    |
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

- **2026-05-21** -- saga 31 step 002: to_number(s), to_int(s),
  env(name) builtins shipped (commit `87f4a2b`). All three return
  Value::Result so callers branch explicitly on failure via
  is_ok / unwrap_or / err_message. Implementation in
  crates/mlpl-eval/src/result_ops.rs (no new modules because
  mlpl-eval is already at the sw-checklist module-count cap).
  Closes audit findings #25 and #26.
- **2026-05-21** -- saga 31 step 001: print(v) / eprint(v) builtins
  shipped (commit `4f7f1f2`). Eval-side dispatch in
  crates/mlpl-eval/src/eval.rs; writes v's Display form to stdout
  / stderr with newline and returns v unchanged so calls compose
  into expressions. Closes audit #28.
- **2026-05-20** -- saga 30 step 006 (FINAL, saga closed): closed
  out audit findings #18 and #19 in `docs/language-audit.md`
  with shipped headers + commit SHAs; moved both findings into a
  new "Shipped" subsection at the top of `docs/plan.md`'s
  Breaking-change candidates. Saga 30 (`tier1-cleanup`) complete.
- **2026-05-20** -- saga 30 step 005: tightened the multi-head pets
  demo intros/takeaways to state concrete accuracy ("training
  accuracy = 1.0 in 30 adam steps") and dropped a "should look
  DIFFERENT" hedge in the attention-overlay intro to "look
  DIFFERENT after training." The existing strings were already
  describing trained behavior accurately (the demo authors were
  not fooled by the stale audit); these refinements just make
  the empirical claim concrete and verifiable.
- **2026-05-20** -- saga 30 step 004: audit finding #19 was stale.
  Empirical verification: `vit_multihead_quick.mlpl` (heads=4,
  100 adam steps, 20 samples) reaches accuracy 1.0; the browser
  config (8 samples, 30 steps) also reaches loss ~0 and accuracy
  1.0. The multi-head tape was already lowered in saga 29 step
  013 (reshape + take + per-head SDPA + `Tensor::stack`); the
  audit was written from an earlier mid-saga-29 state and never
  refreshed. Added a `multi_head_trains_end_to_end_loss_decreases`
  regression test pinning the behavior. The audit finding has
  been rewritten with a SHIPPED status and a "historical claim
  (now refuted)" section.
- **2026-05-20** -- saga 30 step 003: no live workaround to drop;
  the rank-3 attention path already uses `Tensor::stack` (saga 29
  step 008) which is the correct primitive. Cleaned up two stale
  doc comments (`model_tape.rs` module doc saying "chained concat
  over the head axis"; `Tensor::concat` rustdoc saying "0 or 1
  supported in initial release") that described pre-saga-30
  behavior. Added a `[B=2, T=4, d_model=8]` rank-3 single-head
  regression test pinning the shape and per-batch elementwise
  agreement, so any future regression to a chained-binary-concat
  lowering would fail.
- **2026-05-20** -- saga 30 step 002: audit #18 backward lifted.
  The autograd `concat_backward` now generalizes to any axis,
  matching the forward. Rank-3 and rank-4 finite-difference
  gradchecks pass. Closes the audit finding.
- **2026-05-20** -- saga 30 step 001: audit #18 forward lifted in
  `c133d57`. `mlpl-array::concat` now accepts any `axis` in
  `[0, rank)`.
