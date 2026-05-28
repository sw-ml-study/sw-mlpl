# Decomposing the god crates (mlpl-eval, mlpl-web)

This doc captures the strategy for fixing the two largest module-count
FAILs in the codebase: `mlpl-eval` (96 modules) and `mlpl-web` (75
modules). Both are "god crates" -- single workspace members that
centralize an entire feature area instead of spreading its concerns
across multiple sibling crates.

## Why they grew so big

### mlpl-eval (96 modules, by prefix)

| Prefix | Count | Purpose |
|--------|-------|---------|
| `env_*` | 26 | `impl Environment` extensions, one per state category |
| `model_*` | 18 | Model DSL: ~9 ops x 2 layers (`apply_*` + `eval_*`) |
| `eval_*` | 11 | Eval dispatcher per AST node category |
| `fetch_*` | 6 | Dataset download / fetch pipeline |
| `inspect_*` | 5 | `:inspect` / `:describe` introspection |
| `grad_*` | 4 | Autograd-tape integration |
| `fncall_*` | 4 | Built-in function call dispatch |
| `error_*` | 4 | Error type + formatting |
| `tag_*`, `image_*`, `experiment_*` | 6 | Smaller subsystems |
| Singletons | 12 | value, type, device, result, tokenizer, ... |

**Root cause:** `Environment` is the kitchen-sink type. Every operation
hangs methods off it via `impl Environment { fn manage_X() }`. By
Rust's orphan rule, all 26 `env_*.rs` files MUST live in the same
crate as `Environment`. They can't move out without either:

1. Splitting `Environment` itself into smaller types, OR
2. Refactoring all 26 files into extension traits (`EnvFooExt`).

### mlpl-web (75 modules, by prefix)

| Prefix | Count | Purpose |
|--------|-------|---------|
| `demos_*` | 12 | One file per demo (lin reg, MLP, CNN, GAN, ...) |
| `render_*` | 11 | Render branches (header, main, history, completion, ...) |
| `paths_*` | 6 | Learning path content (one file per path) |
| `component_*`, `components.rs` | 7 | Yew components |
| `app_*` | 6 | Top-level app wiring + state |
| `handlers_*` | 5 | Event handlers |
| `onboarding_*` | 4 | Splash + tour + what's-new |
| `viz3d_*`, `mode_*`, `upload_*`, `glossary_*` | 10 | Feature clusters |
| Singletons | 14 | lib, entry, editor, completion, ... |

**Root cause:** `AppState` is the kitchen-sink state, same problem as
`Environment`. Every handler and render branch takes `AppState`
references. Splitting requires either extension-trait pattern or
`AppState` decomposition.

## Why the 5+/-2 guideline doesn't apply directly

The 5+/-2 rule is about cognitive complexity at one zoom level. A
96-module crate isn't a 96-concept abstraction -- it's a 10-concept
feature area where each concept got N files. The right intervention is
structural decomposition at the component/crate level, not file
consolidation. Merging `env_*.rs` files to get under 7 modules would
produce 1000+ line files (FAIL on a different axis).

## Decomposition strategy

The pattern that worked for `mlpl-array` (saga 53): extract operation
families into sibling extension-trait crates within a sub-component,
preserving call-site syntax via `prelude` modules.

### Phase 1: mechanical extractions (no Environment/AppState refactor)

These groups have minimal or no dependency on the kitchen-sink type
and can move out today:

**mlpl-eval:**
- `image_*` (2 files, 0 Environment refs) -> `components/eval-image/`
- `fetch_*` (6 files, 1 Environment ref) -> `components/eval-fetch/`
- `experiment_*` (2 files, 1 ref) -> `components/eval-experiment/`
- `inspect_*` (5 files, 3 refs) -> `components/eval-inspect/` (some
  inspect functions need Environment access; lift to extension trait
  or pass smaller types)

Drops mlpl-eval from 96 to ~80 modules. Still FAIL but big reduction.

**mlpl-web:**
- `demos_*` (12) -> `components/web-demos/`
- `paths_*` (6) -> `components/web-paths/`
- `onboarding_*` (4) -> `components/web-onboarding/`
- `viz3d_*` (3) -> `components/web-viz3d/`

Drops mlpl-web from 75 to ~50 modules. Still FAIL but big reduction.

### Phase 2: trait inversion (the hard lift)

Both `Environment` and `AppState` need to shrink. The fix is the same
pattern that worked for `DenseArray`:

1. Identify which fields/methods truly belong on the core type vs
   which are sub-systems hung off it.
2. For each sub-system, define an extension trait + impl in a sibling
   crate:
   ```rust
   // In mlpl-eval-env-vars (sibling crate):
   pub trait EnvVarsExt {
       fn get_var(&self, name: &str) -> Option<&Value>;
       fn set_var(&mut self, name: String, value: Value);
   }
   impl EnvVarsExt for Environment { ... }
   pub mod prelude { pub use super::EnvVarsExt; }
   ```
3. Callers pick up `use mlpl_eval_env_vars::prelude::*;` -- the
   `env.get_var(...)` call syntax is unchanged.
4. Each extension-trait crate has 1-3 modules of its own. Sparse.

Goal after Phase 2: split mlpl-eval's `env_*` 26 files into ~6 sibling
crates inside `components/eval-env-X/` sub-components, each with 3-5
files.

### Phase 3: model_* and similar dispatch chains

`model_*` (18 files) and `eval_*` (11 files) are dispatch chains where
each file calls others. After Phase 2 lifts `Environment` to a smaller
core, these can move out as sibling crates that take `&mut dyn EnvX`
through the new traits.

## Sub-component grouping (post-decomposition)

After all phases, `mlpl-eval`'s 96 modules become:

```
components/eval-types/        (3)  value, error, env (the core types)
components/eval-image/        (1)  mlpl-eval-image
components/eval-fetch/        (1)  mlpl-eval-fetch
components/eval-experiment/   (1)  mlpl-eval-experiment
components/eval-inspect/      (1)  mlpl-eval-inspect

components/eval-env-vars/     (1)  EnvVarsExt
components/eval-env-models/   (1)  EnvModelsExt
components/eval-env-tensors/  (1)  EnvTensorsExt
components/eval-env-devices/  (1)  EnvDevicesExt
components/eval-env-strings/  (1)  EnvStringsExt
components/eval-env-misc/     (1)  EnvDirsExt, EnvParamsExt, etc.

components/eval-model-apply/  (3)  model_apply, model_apply_*
components/eval-model-eval/   (3)  model_eval_*, model_dispatch
components/eval-model-lora/   (1)  model_lora, model_mutate
components/eval-model-inspect/(1)  model_freeze, model_feasibility

components/eval-dispatch/     (3)  eval_*, fncall_*
components/eval-grad/         (1)  grad_*
components/eval-tag/          (1)  tag_propagate, auto_tag
components/eval/              (1)  facade re-exporting everything
```

That's ~18 sibling crates across ~12 sub-components, each sparse.

For `mlpl-web` similarly:

```
components/web-content/       (3)  demos, paths, glossary
components/web-onboarding/    (1)  splash + tour + what's-new
components/web-viz3d/         (1)  3D scene + handlers
components/web-render/        (3)  render_* split by area
components/web-components/    (3)  Yew components split by concern
components/web-handlers/      (2)  event handlers
components/web-state/         (2)  AppState decomposed
components/web/               (1)  app entry + lib facade
```

## Migration plan (when ready)

1. **One saga per Phase-1 extraction.** Each is mechanical (~30 min)
   and drops 2-12 modules. Total: ~8 sagas, drops eval to ~80 and
   web to ~50.
2. **One saga per Environment / AppState sub-system trait inversion.**
   Each is bigger (~1-2 hours) but high payoff. Total: ~6 sagas, drops
   eval to ~25 modules across the env-X siblings.
3. **One saga per model_* / eval_* dispatch chain.** Total: ~4 sagas.
4. **Mirror approach for mlpl-web.** Total: ~6 sagas.

Estimated 24 sagas total to fully decompose both god crates. Each
ends with `cargo test` green and the sw-checklist count moving in the
right direction.

## Why not just refactor in place

A few alternatives considered and rejected:

- **Merge files to fewer modules.** Trades module-count FAIL for
  file-LOC FAIL. Worse from a code-review perspective.
- **Suppress sw-checklist for these crates.** Defeats the purpose;
  the project standard is uniform.
- **Move everything into a "god component" with many crates.** Just
  shifts the sprawl up a level (see also `feedback_4_per_workspace`
  in memory).

The extension-trait pattern is the only path that preserves call-site
syntax, retires the FAILs, and keeps modules sparse at every zoom
level.

## Status

- Phase 0 (the bottom-up component migration) is **complete** as of
  saga 67-70: 33 sub-components live, lower-layer crates all under the
  4-crate-per-workspace cap.
- Phase 1 / 2 / 3 are queued for future sagas.
