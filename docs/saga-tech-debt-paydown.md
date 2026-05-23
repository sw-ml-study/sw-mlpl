# Saga 32: tech-debt paydown

## Why this exists

`sw-checklist` reports 151 FAIL and 452 WARN entries at the start
of this saga. Saga 31 (scripting-cluster) shipped nine audit
findings in eight steps and retired -6 fails / -5 warnings -- a
trickle compared to the per-commit mandate in `CLAUDE.md` (target
-5 fails AND -15 warnings per commit, or halve both during a
dedicated tech-debt spike). The structural problem is that
in-place compression can only nibble at the warning column;
the FAIL column is dominated by **crate / module / file
structural budgets** that demand splits, not one-line tweaks.

Top FAIL breakdown (151 total):

- 70 -- Module Function Count (modules with > 7 fns)
- 32 -- Function LOC (fns over 50 lines)
- 28 -- File LOC (files over 500 lines)
- 12 -- Clippy Allows (`#[allow(...)]` attributes outside vendored code)
-  8 -- Crate Module Count (crates with > 7 modules)
-  1 -- Rust Edition

Eight crates are over the 7-module Crate-Module-Count cap. The
worst offenders:

- `mlpl-eval`: **42 modules** (max 7)
- `mlx-rs`:    79 modules (vendored; addressed by vendoring policy, not by splitting)
- `mlpl-web`:  29 modules
- `mlpl-runtime`: 18 modules
- `mlpl-parser`, `mlpl-serve`, `mlpl-repl`, `mlpl-viz`: 9-10 modules each

The cascade effect: every fat crate also racks up Module-Fn-Count
fails (each of its modules has too many fns) and File-LOC fails
(modules grow to absorb the workload). Splitting one fat crate
into 3-4 sibling crates retires:

- 1 Crate-Module-Count FAIL
- ~5-15 Module-Fn-Count FAILs (functions get spread across more
  modules)
- ~5-10 Function-LOC FAILs (the new modules give room for
  responsibility-bounded helpers that were impossible inside
  the fat module)

That is the structural-payoff ratio per `CLAUDE.md`'s
"refactoring saga that splits one fat crate can clear many at
once."

## Goals

- **Halve the FAIL count.** 151 -> ~75. Achievable via the
  three big crate splits alone, plus the Function-LOC sweep
  that opens up once the new modules exist.
- **Halve the warning count.** 452 -> ~225. Each crate split
  retires the long-tail of "this fn is 26 lines because there
  is no other place to put the extracted helper" warnings.
- **No new features.** Source behaviour does not change. Every
  test that passes today must pass after every step.
- **Each step net-negative on BOTH fails AND warnings.** No
  exceptions; if a step can't beat the prior baseline, the
  scope is wrong.

## Non-goals

- Touching `mlx-rs` (vendored, addressed via `vendor/` policy).
- Changing public API surface. Every `pub use` in a crate's
  `lib.rs` keeps the same import path. (If a crate splits in
  two, the old crate becomes a facade that re-exports.)
- Hunting for new clippy lints. Existing FAIL count is the
  scoreboard.

## Dependencies

- No new dependencies. May TRIM existing dependencies where a
  split lets a downstream crate drop a feature it never used.

## Canonical HOW-TO

For the techniques used to refactor wide functions and split
fat modules, see [`docs/loose-coupling.md`](loose-coupling.md).
That document captures the four phases (compile time,
start-up, conditional, dataflow) and the
compose-don't-compress patterns. Every step in this saga
applies them.

## Quality requirements (every step)

Same as saga 31's strict variant:

1. RED/GREEN/REFACTOR if any logic changes (most steps are
   refactors only -- no new tests needed, but every existing
   test must still pass).
2. `cargo test --workspace` green.
3. `cargo clippy --workspace --all-targets --all-features -- -D warnings` green.
4. `cargo fmt --all -- --check` green.
5. `markdown-checker -f "**/*.md"` green for any touched docs.
6. **`sw-checklist` must net-negative on BOTH fails AND
   warnings vs the previous commit.** Each step's commit body
   must quote the before/after counts.
7. Push after every commit.

## Steps

### Step 001 -- split mlpl-eval (42 modules -> 3-4 sibling crates)

`crates/mlpl-eval/` has 42 modules and is the biggest single
source of structural debt. Triage the modules by responsibility:

- **`mlpl-eval` (eval core)** -- `eval.rs`, `eval_program.rs`,
  `eval_for.rs`, `eval_loop.rs`, `eval_intercepts.rs`,
  `eval_ops.rs`, `eval_reduce.rs`, `eval_script.rs`, `env.rs`,
  `error.rs`, `value.rs`, `device.rs`, `interrupt.rs`. The
  expression-walker and its closest helpers stay here.
- **`mlpl-eval-model` (new sibling crate)** -- `model.rs`,
  `model_clone.rs`, `model_dispatch.rs`, `model_embed_table.rs`,
  `model_estimate.rs`, `model_feasibility.rs`, `model_freeze.rs`,
  `model_lora.rs`, `model_perturb.rs`, `model_tape.rs`. Every
  `ModelSpec` operation. 10 modules clearly bundled by topic.
- **`mlpl-eval-grad` (new sibling crate)** -- `grad.rs`,
  `grad_optim.rs`, `tag_propagate.rs`, `tag_render.rs`,
  `auto_tag.rs`. Autograd + gradient accumulation.
- **`mlpl-eval-data` (new sibling crate)** -- `bpe.rs`,
  `loader.rs`, `tokenizer.rs`, `type_errors.rs`,
  `metric_sink.rs`, `experiment.rs`, `inspect.rs`,
  `inspect_groups.rs`, `result_ops.rs`, `llm_dispatch.rs`,
  `pets_tiny.rs`, `fetch_dataset.rs`, `image_io.rs`. Loaders,
  tokenizers, telemetry.

`mlpl-eval`'s `lib.rs` becomes a facade that `pub use`s every
existing public name from the new crates so downstream crates
(`mlpl-cli`, `mlpl-serve`, `mlpl-repl`, etc.) import unchanged.
The seam is purely structural.

Expected retirement: 1 Crate-FAIL (mlpl-eval drops from 42 to
~13 modules) + ~5-10 Module-Fn-Count FAILs + ~3-5 Function-LOC
FAILs as the new crates give room for helpers.

### Step 002 -- split mlpl-runtime (18 modules)

`crates/mlpl-runtime/` has 18 modules organized as a big
builtins registry. Split by responsibility:

- **`mlpl-runtime` (core)** -- the dispatch registry and the
  small subset that everything else depends on (call_builtin,
  RuntimeError, BUILTINS slice).
- **`mlpl-runtime-math` (new)** -- math + comparison + reduce
  builtins (`gt`, `lt`, `exp`, `log`, `sqrt`, `sigmoid`,
  `tanh`, `sum`, `prod`, `mean`, `max`, `min`).
- **`mlpl-runtime-array` (new)** -- shape + indexing + linear
  algebra builtins (`reshape`, `transpose`, `iota`,
  `dot`, `matmul`, `zeros`, `ones`).
- **`mlpl-runtime-ml` (new)** -- attention / softmax / one-hot
  / token-batch / image-decode builtins (anything that's
  ML-shaped).

Same facade pattern: `mlpl-runtime`'s `lib.rs` re-exports
everything.

Expected retirement: 1 Crate-FAIL (mlpl-runtime drops from 18
to ~6 modules) + ~4-8 Module-Fn-Count FAILs.

### Step 003 -- split mlpl-web (29 modules)

`apps/mlpl-web/` has 29 modules covering REPL UI, tutorial,
paths, viz cache, eval-over-SSE, etc. Split:

- **`mlpl-web` (UI shell)** -- entry, layout, top-level
  components.
- **`mlpl-web-tutorial` (new)** -- tutorial panel + paths
  view + intro markdown.
- **`mlpl-web-eval` (new)** -- eval session, SSE streaming,
  viz cache, paths_view, upload.

Facade keeps `apps/mlpl-web` as the binary that pulls the
library crates together.

Expected retirement: 1 Crate-FAIL + ~5-8 Module-Fn-Count FAILs.

### Step 004 -- Function LOC sweep (32 -> ~10 FAILs)

With the new modules in place from steps 001-003, walk the
remaining 32 Function-LOC FAILs and extract helpers. Each
extracted helper goes into a sibling module (not a new fn
inside the over-budget module). Use `docs/code_metrics.md`'s
file-naming convention: `parse.rs` for input -> typed,
`validate.rs` for typed -> result, `render.rs` for data ->
string, etc.

Target: -15 Function-LOC FAILs, -20 Function-LOC warnings.

### Step 005 -- File LOC sweep (28 -> ~10 FAILs)

Same approach for the 28 File-LOC FAILs (files over 500
lines). Most are now in the new crates from steps 001-003;
the rest need responsibility-bounded splits per
`docs/code_metrics.md`.

Target: -15 File-LOC FAILs.

### Step 006 -- Clippy Allows audit (12 -> 0 FAILs)

12 `#[allow(clippy::...)]` attributes outside `mlx-rs`. Each
one needs either a real fix (refactor the code to satisfy the
lint) or a documented justification with a TODO + targeted
ticket. Default: fix.

Target: -10 to -12 Clippy-Allows FAILs.

### Step 007 -- warning long-tail sweep

By this step the FAIL column should be in the 70-90 range.
Pivot to warnings: walk the 26-30 line Function-LOC warnings
and the 350-500 line File-LOC warnings, extracting helpers
or splitting files per `docs/code_metrics.md`. Avoid struct-
literal compressions that rustfmt will revert.

Target: -50 warnings.

### Step 008 -- final ratchet + update docs

Final sw-checklist pass + verify all gates green. Update
`docs/code_metrics.md`'s motivating example if needed. Refresh
`CHANGES.md`. Update `docs/language-status.md` with the
saga-close entry.

Target: final state in the 70-90 fails / 200-250 warnings
range. Halved on both axes vs the 151/452 starting point.

`agentrail complete --done`.
