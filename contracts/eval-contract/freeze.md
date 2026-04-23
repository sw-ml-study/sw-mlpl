# `freeze` / `unfreeze` Contract (Saga 15 step 001)

## Purpose

`freeze(m)` marks every parameter of a model as "frozen": the
parameters stay trainable in the sense that `grad` still
produces gradients for them (the chain rule does not care
about freeze state), but `adam` and `momentum_sgd` skip the
parameter-update step for any name in the frozen set. The
weight stays put while the rest of the model trains.

`unfreeze(m)` is the inverse. After a `freeze` / `unfreeze`
pair, the model is back to its fully-trainable state.

This is the groundwork for Saga 15's LoRA fine-tune: the user
`freeze(base)`s the pre-trained backbone, then
`lora(base, rank, alpha, seed)` adds small trainable adapter
matrices, and `adam` moves only the adapters.

## Signature

```
freeze(m)   -> scalar 0.0
unfreeze(m) -> scalar 0.0
```

- `m` -- a model identifier (bare `Expr::Ident` bound in
  `env.models`) or any expression that evaluates to
  `Value::Model` (e.g. `linear(...)`, `chain(...)`).
- Return value is a scalar-zero unit, mirroring
  `to_device` / `perturb_params` / every other in-place
  model builtin. Lets the call sit in statement position.

## Semantics

- `freeze(m)` inserts every name in `m.params()` into
  `env.frozen_params`. Idempotent: freezing an
  already-frozen model is a no-op.
- `unfreeze(m)` removes every name in `m.params()` from
  `env.frozen_params`. Idempotent: unfreezing a non-frozen
  model is a no-op.
- `env.is_frozen(name)` exposes the state for inspection
  and for the optimizer filter.
- `adam(loss, frozen_m, ...)` and `momentum_sgd(loss,
  frozen_m, ...)` iterate `frozen_m.params()` as before but
  `continue` for every name in the frozen set. No gradient
  is computed for frozen names; no optimizer state
  (`m` / `v` buffers) is updated; the stored weight is
  bit-identical after N training steps.
- Grad computation is unchanged. `grad(expr, frozen_W)`
  still returns the correct gradient; downstream ops that
  depend on `frozen_W` still see its weight.

## Error cases

- **Wrong arity.** Anything other than 1 arg returns
  `EvalError::BadArity { func: "freeze" | "unfreeze",
  expected: 1, got }`.
- **Non-identifier that does not evaluate to a Model.**
  `EvalError::Unsupported("freeze: argument must evaluate
  to a model")`.
- **Bare identifier that is not a model.** If the user
  passes `freeze(x)` where `x` is an array var, the lookup
  into `env.models` fails and surfaces as
  `EvalError::Unsupported("freeze: 'x' is not a model")`.

## What this contract does NOT cover

- **Per-parameter freezing.** Saga 15 ships whole-model
  freeze. Freezing individual names (e.g. "freeze W but
  not b") is a follow-up; a likely API shape is
  `freeze(m, "__linear_W_5")` or a regex/family-style
  filter that reuses the Saga 20 family walker.
- **Freezing a model makes its gradients zero.** It does
  not. Freeze is strictly optimizer-side; the autograd tape
  is unchanged.
- **Checkpoint-save / re-freeze across sessions.** `env`
  state is in-memory only today; persisting the frozen
  set to disk is part of the future checkpoint saga.
- **Interaction with `clone_model`.** Cloning a model does
  NOT propagate frozen status: the clone has fresh param
  names and those names are not in `env.frozen_params`
  unless the caller explicitly calls `freeze` on the clone.
  Saga 15 step 002's `lora(m, ...)` wraps base linears
  directly rather than relying on clone-time freeze
  propagation.
