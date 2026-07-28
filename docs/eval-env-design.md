# mlpl-eval-env design -- unblocking the eval decomposition

Design doc required by the eval-decomposition saga before touching
the `env_*` cluster. Written 2026-07-28 after the models-out step
hit the coupling wall described below.

## The finding that reorders the saga

Every extractable cluster in `mlpl-eval` -- `model_*` (19),
`fetch_*` (6), `inspect_*` (6), `grad_*` (6), `device_*`/`gpu_*`
(6) -- takes `&mut Environment`. `Environment`, `eval_expr`, and
`device::dispatched_call` all live in the hub crate, and the
clusters' consumers (fncall dispatch, auto_tag) also live in the
hub. A sibling crate therefore cycles: it needs `Environment` from
`mlpl-eval` while `mlpl-eval` needs the cluster's functions.
**models-out (and every other cluster peel) is blocked until the
env layer moves below the hub.** The original step order was
optimistic; env-out is the prerequisite, not the finale.

## Why one env crate cannot be checklist-clean

The env cluster is `env.rs` + 27 `env_*` modules, ~1045 lines, **90
functions**, almost all inherent `impl Environment` extensions.
Inherent impls must live in the defining crate, so:

- One crate holding all of them has 28 modules (crate FAIL at >7).
- Consolidating into <= 7 files needs <= 49 functions (7 files x 7
  fn cap); 90 do not fit.

So neither "move as-is" nor "move and squash" lands inside budgets.

## The design: capability traits (the mlpl-env-traits pattern)

The escape is already in the codebase: `mlpl-env-traits`
(session-infra) defines `HasVars`, `HasModels`, `HasParams`,
`HasDispatch`, ... and `env_trait_impls_*.rs` implement them on
`Environment`. Generalize that:

1. **Base crate `mlpl-eval-env`** (eval workspace): the
   `Environment` struct, its fields (made `pub`, documented as the
   eval component family's internal API), `new()`/constructors, and
   the handful of invariant-preserving methods that must stay
   inherent. Small: <= 5 modules.
2. **Capability crates** `mlpl-eval-envc-*` (bindings, mlstate,
   exec, obs): each env_* module becomes a `trait EnvXyz` +
   `impl EnvXyz for Environment` pair in its own file. The orphan
   rule allows the impl in the TRAIT's crate even though
   `Environment` is foreign. Each trait keeps <= 7 methods (one
   trait per old env_* module maps almost 1:1), each crate holds
   3-6 such modules -- all budgets hold arithmetically:
   90 fns / 7-per-module = 13+ modules across 4 crates.
3. **Call sites keep method syntax.** `env.get_var(...)` still
   compiles wherever the trait is in scope; consumers add one
   `use` line. `mlpl-eval` re-exports every capability trait from
   an `env_api` prelude module so its ~60 consumer files just say
   `use crate::env_api::*;`.

### Leaf state types (move first, no controversy)

`Environment` fields reference six leaf types that must move below
first -- each is either a whole small module or the type-half of a
mixed module (type moves, eval handlers stay):

| Type | Today | Move |
| --- | --- | --- |
| `TokenizerSpec` | tokenizer.rs (mixed, 169 ln) | type-half down |
| `ExperimentRecord` | experiment.rs (mixed, 175 ln) | type-half down |
| `GpuAdamStep` | gpu_step.rs (mixed, 64 ln) | trait/type down, mlx registration stays |
| `GpuEnv` | gpu_env.rs (22 ln, dep-free) | whole |
| `OptimizerState` | grad.rs (struct at ~line 230) | struct down |
| `Interrupt` | interrupt.rs (81 ln) | whole |

Destination: `mlpl-eval-core` (components/types) -- `ModelSpec` and
`MetricSink` already live there, and `Value`/`EvalError` are
already below in `mlpl-eval-types`, so precedent is established.
`gpu_registry.rs` (39 ln, OnceLock-style registry) moves with
`GpuAdamStep`; the hub keeps installing its default steps.

### The two function seams

- `dispatched_call` (device.rs, calls back into eval): only used by
  `env_trait_impls_dispatch.rs`, which is a `HasDispatch` trait
  impl -- it can move to the base crate and route through a
  hub-installed hook.
- `eval_expr` (needed later by model_*/device_* peels): same
  pattern -- `mlpl-eval-env` exposes a `OnceLock` hook
  (`EvalFn = fn(&mut Environment, &Expr) -> Result<Value, EvalError>`;
  `Expr` comes from mlpl-parser, already below). `mlpl-eval`
  installs it at evaluator entry. Mirrors the existing
  `gpu_registry::installed_gpu_step` inversion.

## Corrected step order

1. **env-types-out** -- the six leaf types (table above) to
   mlpl-eval-core; mechanical, no behavior change.
2. **env-base-out** -- `Environment` + all `env_*` files move to
   `mlpl-eval-env` as-is. TRANSIENT: the new crate is over module
   budget until step 4 completes; each commit carries a documented
   `sw-checklist: exception` naming this plan.
3. **env-capability-peels** (one or more commits) -- convert each
   env_* module to trait+impl in its `mlpl-eval-envc-*` crate;
   `env_api` prelude grows as they land.
4. **transient ends** -- mlpl-eval-env down to <= 5 modules.
5. **cluster peels, now unblocked** -- models-out (19), fetch-out,
   inspect-out, device-out (installs the eval/dispatch hooks),
   grad-out, in the original saga's shapes.
6. **spine-tidy** -- the hub lands at <= 7 modules.

## Constraints carried forward

- Facade-preserving: `mlpl_eval::Environment` etc. stay valid via
  `pub use` through every step.
- Scoped tests per step: eval workspace `--release` for interpreter
  suites + cli/serve/web dependents.
- Post-rustfmt budgets; no `#[allow]`; the transient exception in
  steps 2-3 is the ONLY sanctioned one and must shrink each commit.
