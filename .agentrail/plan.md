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

## Saga steps (draft)

1. **models-out** -- `mlpl-eval-models` (19 modules out, one-way
   deps, biggest single win). Facade re-exports hold paths.
2. **fetch-out** -- `mlpl-eval-fetch` (6 modules + the 4 fetch
   test-helper modules become that crate's tests/).
3. **inspect-out** -- `mlpl-eval-inspect` (6).
4. **grad-out** -- `mlpl-eval-grad` (6; watch the Tape/params
   plumbing through eval_blocks).
5. **device-out** -- `mlpl-eval-device` (6; mlx/cuda feature
   pass-through moves with it).
6. **env-capabilities** -- the hard step: regroup 28 `env_*`
   modules into <= 5 capability components owned by
   `Environment` (composition over inherent-impl sprawl).
   Requires its own design doc before execution.
7. **spine-tidy + wrap** -- the hub lands at <= 7 modules
   (lib, env, eval spine, fncall dispatch, result_ops, ...);
   full-suite release-profile run; counts recorded.

Estimated: each of steps 1-5 is one session (the serve split
took one); step 6 is two-plus. The interpreter's 900+ test
suite runs `--release` per the disk-aware build rules.

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