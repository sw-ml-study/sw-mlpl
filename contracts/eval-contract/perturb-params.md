# `perturb_params` Contract (Saga 20 step 002)

## Purpose

`perturb_params(m, family, sigma, seed)` adds family-targeted
Gaussian noise to a model's parameters in place. Paired with
`clone_model` (step 001) it expresses the core Neural Thickets
move: one base, N independently perturbed variants.

## Signature

```
perturb_params(m, family, sigma, seed) -> scalar (0.0)
```

- `m` -- model identifier (bare `Expr::Ident` bound in
  `env.models`). Non-identifier first args are rejected.
- `family` -- string literal from the accepted set.
- `sigma` -- scalar f64 (any value; `sigma == 0.0` is a
  lawful no-op, negative `sigma` is mathematically fine and
  passes through unmodified).
- `seed` -- scalar f64 used as the base PRNG seed.
- Returns `DenseArray::from_scalar(0.0)` so the call can sit
  in statement position or be consumed as a scalar, mirroring
  `to_device`'s unit-return convention.

## Accepted families

- `all_layers` -- every parameter owned by the model.
- `attention_only` -- names starting with `__attn_` (the four
  projection matrices `Wq`, `Wk`, `Wv`, `Wo` allocated by
  `attention` / `causal_attention`).
- `mlp_only` -- names starting with `__linear_` EXCEPT the
  parameters of the final projection head (see below).
- `embed_and_head` -- names starting with `__embed_` PLUS the
  final projection head's parameters.

## Structural "final projection head" rule

The head is defined structurally, not by a name pattern:

- If `m` is a bare `Linear`, it IS the head.
- If `m` is a `Chain`, the head is the last direct child that
  is a `Linear` variant (searched right-to-left over
  top-level children; nested linears inside a `Residual` or a
  nested `Chain` do NOT count).
- Otherwise (a `Residual`, a bare `Attention`, etc., wrapping
  the whole model), there is no head. `mlp_only` then touches
  every `__linear_*` parameter; `embed_and_head` touches only
  `__embed_*` parameters.

This rule is deliberately not a name-only heuristic. Name
patterns cannot distinguish the last MLP layer of a
transformer block from the final vocab projection -- both are
`__linear_*`. Structural position is the only unambiguous
signal the Model DSL currently offers.

## Seeding + determinism

- Affected parameters are walked in `ModelSpec::params()`
  order (the same order `grad`, `adam`, and
  `momentum_sgd` see).
- The i-th affected parameter is perturbed with the PRNG
  seed `seed + i as f64`, so same-shape parameters get
  independent deltas but two clones of the same source with
  the same `(family, sigma, seed)` produce bit-identical
  deltas.

## Magnitude

Internally the perturbation is
`param = param + sigma * randn(seed_i, shape(param))`.
`randn` is a Box-Muller sample on a 53-bit xorshift uniform,
so absolute delta magnitudes are practically bounded by
roughly `10 * sigma` per element.

## Error cases

- **Wrong arity.** Anything other than 4 args returns
  `EvalError::BadArity { func: "perturb_params", expected:
  4, got }`.
- **Non-identifier first argument.** Returns
  `EvalError::Unsupported("perturb_params: first argument
  must be a model identifier")`.
- **Non-model identifier.** Returns
  `EvalError::Unsupported("perturb_params: '<name>' is not a
  model")`.
- **Non-string `family`.** Returns
  `EvalError::Unsupported("perturb_params: family (second
  argument) must be a string literal")`.
- **Unknown family string.** Returns
  `EvalError::Unsupported("perturb_params: unknown family
  '<name>' (expected one of all_layers, attention_only,
  mlp_only, embed_and_head)")`.
- **Non-scalar `sigma` or `seed`.** Returns
  `EvalError::Unsupported("perturb_params: sigma and seed
  must be scalars")`.

## What this contract does NOT cover

- Low-rank perturbation (`perturb_low_rank`). Sibling builtin
  deferred to a follow-up; Saga 20 ships the Gaussian case
  only.
- Depth-aware families (`early_N_layers`, `late_N_layers`).
  Require explicit layer indices in param names, which the
  Model DSL does not encode today.
- Compile-to-Rust (`mlpl-rt`) parity. `perturb_params` is
  interpreter-only in Saga 20; port if future parity tests
  require it.
- Autograd. Perturbation is an evaluation-time, non-
  differentiable op. Gradients do not flow through it.
