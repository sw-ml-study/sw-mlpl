# mlpl-eval decomposition saga (plan)

Status: PLANNED -- the one remaining sw-checklist FAIL after the
2026-07 tech-debt spike (18 failed -> 1). mlpl-eval has 100
modules against the 7-module crate budget; this is a dedicated
saga, not a within-spike refactor, because the crate is the
evaluator's hub and every workspace compiles through it.

## Why it is its own saga

The spike's other seven FAILs each split along an existing seam
(viz marks vs analysis, autograd tape vs tensors, serve
core/state/handlers, repl connect cluster, web-eval floors). The
eval crate's 100 modules interlock through `Environment` -- 28
`env_*` modules are inherent-impl extensions on one struct --
so a mechanical split would either move `Environment` (breaking
every consumer) or leave the hub crate over budget anyway.
Splitting it well means deciding what `Environment` IS first.

## Cluster inventory (2026-07-27)

| Cluster | Modules | Split shape |
| --- | --- | --- |
| `env_*` | 28 | The hub. Facade-preserving sub-crates keyed by capability: vars/records/strings (bindings), models/tokenizers/params (ML state), device/gpu/peer/interrupt (execution context), metric-sink/exp-log (observation). `Environment` stays here; capability clusters become traits or component structs it owns. |
| `model_*` | 19 | Cleanest first cut: model apply/inspect/attention lowering already depends one-way on env. Sibling crate `mlpl-eval-models`. |
| `eval_*` | 11 | The dispatch spine (eval, eval_fncalls, eval_blocks, eval_loop, ...). Stays in the hub crate. |
| `fetch_*` | 6 | Dataset fetch + its test helpers. Sibling crate `mlpl-eval-fetch` (io-shaped, no tape deps). |
| `inspect_*` | 6 | `:vars`/`:describe`/workspace snapshots. Sibling `mlpl-eval-inspect`. |
| `grad_*` | 6 | grad()/train lowering onto mlpl-autograd. Sibling `mlpl-eval-grad`. |
| `fncall_*` + dispatch | 4 | Stays with the spine. |
| `gpu_*`/`device*` | 6 | Device dispatch + registry. Sibling `mlpl-eval-device`. |
| misc (tag, experiment, result_ops, tokenizer, loader, llm, bpe, ...) | ~14 | Distribute with their consumers; result_ops/tag stay in the hub. |

## Saga steps (REVISED 2026-07-28 -- see docs/eval-env-design.md)

The original order (models first, env last) proved impossible: a
models-out attempt found that EVERY cluster takes
`&mut Environment` and every cluster's consumers live in the hub,
so any sibling crate cycles until the env layer moves below.
Corrected order (full rationale + the capability-trait design in
`docs/eval-env-design.md`):

1. **env-types-out** -- six leaf state types (TokenizerSpec,
   ExperimentRecord, GpuAdamStep/GpuEnv/registry, OptimizerState,
   Interrupt) down to mlpl-eval-core.
2. **env-base-out** -- `Environment` + env_* to a new
   `mlpl-eval-env` crate as-is (transient over-budget, documented
   exception until step 3 completes).
3. **env-capability-peels** -- each env_* inherent-impl module
   becomes a `trait EnvXyz` + `impl for Environment` pair in an
   `mlpl-eval-envc-*` capability crate (the mlpl-env-traits
   pattern; orphan rule allows impl-in-trait-crate). Call sites
   keep method syntax via a `use crate::env_api::*;` prelude.
4. **models-out** -- `mlpl-eval-models` (19 modules), with
   `eval_expr` reached through the mlpl-eval-env hook.
5. **fetch-out** -- `mlpl-eval-fetch` (6 + 4 test helpers).
6. **inspect-out** -- `mlpl-eval-inspect` (6).
7. **device-out** -- `mlpl-eval-device` (6; installs the
   dispatch hook).
8. **grad-out** -- `mlpl-eval-grad` (6; watch Tape/params
   plumbing through eval_blocks).
9. **spine-tidy + wrap** -- the hub lands at <= 7 modules;
   full-suite release-profile run; counts recorded.

Estimated: steps 1 and 4-8 are one session each; steps 2-3 are
two-plus combined. The interpreter's 900+ test suite runs
`--release` per the disk-aware build rules.

## Constraints carried from the spike

- Facade-preserving splits only: `mlpl_eval::X` paths hold via
  `pub use`; the `$crate::ops::`-style macro paths need module
  re-exports (see the autograd split precedent).
- Post-rustfmt budgets; no `#[allow]`, no compression tricks.
- New at-cap crates add structural module-count warnings; plan
  one extra offset retirement per step (the spike's recurring
  lesson, documented in three step summaries).
- Scoped tests: eval + dependents (cli, serve, web harnesses)
  only; `--release` for the interpreter-loop suites.
