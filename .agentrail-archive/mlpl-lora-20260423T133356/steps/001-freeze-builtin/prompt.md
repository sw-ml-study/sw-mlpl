Phase 1 step 001: `freeze(m)` + `env.frozen_params`.

Groundwork for LoRA. Adds a second param set to the
evaluator so optimizers can skip a subset of trainable
names (the base weights we are about to LoRA-wrap).

1. Add `frozen_params: HashSet<String>` to `Environment`
   (sibling of the existing `params`). A parameter name can
   be in both sets simultaneously; it still has a value and
   `grad` still produces gradients, but `adam` and
   `momentum_sgd` must skip any name in `frozen_params`
   when applying parameter updates.
2. New `freeze(m)` builtin. Walks `m.params()` and adds
   every name to `env.frozen_params`. Returns a scalar
   zero unit (matching the `to_device` / `perturb_params`
   convention for in-place model-mutating ops). Accept the
   same model-identifier-or-expression argument shape as
   `perturb_params`.
3. New `unfreeze(m)` builtin: the inverse, removing every
   `m.params()` name from `frozen_params`. Decide during
   TDD whether `unfreeze` ships in step 001 or is deferred.
4. Update `crates/mlpl-eval/src/grad.rs` so the adam +
   momentum_sgd param-update loops filter through
   `env.frozen_params`. Grad itself still computes
   gradients for frozen params (the chain rule needs them
   for downstream computations); only the *optimizer
   update* skips them.
5. Contract file
   `contracts/eval-contract/freeze.md`: semantics, the
   "grad still flows, optimizer skips" rule, error cases,
   non-goals (no nested freezing yet; no per-param freeze,
   only whole-model).
6. TDD (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-eval/tests/freeze_tests.rs`:
   - After `freeze(m)`, every `m.params()` name is in both
     `env.params` and `env.frozen_params`.
   - `adam(loss, frozen_m, ...)` across 3 training steps
     leaves every param value bit-identical.
   - Partial freeze: `freeze(m1)` then train `m2` --
     `m1`'s params do not move, `m2`'s do.
   - Gradient sanity: `grad(expr, frozen_m)` produces
     non-zero gradients where expected (freeze does not
     zero gradients).
   - If unfreeze ships: round-trip test
     (freeze -> train no-op -> unfreeze -> train moves).
   - Wrong arity / non-model argument / etc.
7. Module placement: put the new builtin in a small new
   module if `model_dispatch.rs` is already over budget
   (it was at 647 LOC / 10 fns going into Saga 20). Candidates:
   `model_freeze.rs` alongside `model_clone.rs` and
   `model_perturb.rs`, or reuse `model_perturb.rs` since the
   patterns are similar. Pick during implementation.
8. Quality gates + `/mw-cp`. Commit message references
   Saga 15 step 001.
