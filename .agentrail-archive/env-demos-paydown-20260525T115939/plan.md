# Saga 33: env.rs and demos.rs paydown

## Why this exists

The two biggest structural-debt items carried over from saga
32 are:

- **`crates/mlpl-eval/src/env.rs`** (55 methods on the
  `Environment` struct, 498 lines). The single largest
  Module-Function-Count FAIL in the workspace by far. The
  struct conflates start-up construction, per-call mutation,
  and per-item lookup -- the classic phase-conflation
  anti-pattern that `docs/loose-coupling.md` flags as the
  highest-value refactor target.

- **`apps/mlpl-web/src/demos.rs`** (1179 lines, 29 inline
  `Demo` consts in a single `&[Demo]` array). The largest
  single-file FAIL in the workspace. Pure compile-time data
  that should live in `const` tables organized by topic.

Saga 32 retired -8 fails and -2 warnings; the two files
above account for the next ~10-15 fails available at "easy
to retire if you split by phase" pricing. They are the
right target for the follow-up saga.

## Approach

Apply the techniques in [`docs/loose-coupling.md`](loose-coupling.md):

- **demos.rs is a phase-1 (compile-time) refactor.** Split
  by topic into sibling files; each file holds ~5 `pub
  const FOO: Demo = Demo { ... };` declarations and is
  150-250 lines. `demos.rs` becomes a thin facade with the
  `Demo` struct, the `PROGRESS_NOTES` table, the
  `progress_notes_for` helper, and a `pub const DEMOS:
  &[Demo] = &[<refs>]` array referring to the topical
  consts.

- **env.rs is a phase-by-responsibility refactor.** The
  `Environment` struct stays in `env.rs`; each topical
  group of methods moves into a sibling `impl Environment`
  block in its own file (`env_vars.rs`, `env_models.rs`,
  ...). Rust allows split impl blocks across modules in
  the same crate, so callers and the struct layout are
  unchanged. The struct continues to own all the data
  fields; the sibling files only host method bodies.

## Goals

- Retire `env.rs` Module-Function-Count FAIL (55 -> <=7).
- Retire `demos.rs` File-LOC FAIL (1179 -> <500).
- No new FAILs introduced. New sibling modules in `mlpl-eval`
  are "free" for Crate-Module-Count (already FAILing).
  `apps/mlpl-web` Crate-Module-Count is already FAILing too;
  adding topical demo files is also free for that metric.
- Each new module passes its own fn-count + file-LOC
  budgets.
- Strict DAG preserved.

## Non-goals

- No new builtins, no new language features.
- No semantic change to `Environment`'s public surface;
  callers see the same struct + methods.
- No reorganization of `Demo`'s schema; only the array
  contents move.

## Quality requirements (every step)

Same strict gate as saga 32:

1. `cargo test --workspace` green.
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings` green.
3. `cargo fmt --all -- --check` green.
4. `markdown-checker` green for any touched docs.
5. **`sw-checklist` must net-negative on BOTH fails AND
   warnings vs the previous commit.** Each commit body
   quotes the before/after counts.
6. Push after every commit.

## Steps

### Step 001 -- split demos.rs by topic

Move the 29 `Demo` consts out of `demos.rs` into topical
sibling files in `apps/mlpl-web/src/`:

- `demos_basics.rs`: arithmetic, arrays, broadcasting, reduce
  (the first ~8 entries).
- `demos_models.rs`: linear, chain, residual, attention,
  embed, lora (~6 entries).
- `demos_training.rs`: train loops, loss curves, autograd
  walk-throughs (~5 entries).
- `demos_viz.rs`: histograms, scatter, decision boundary,
  embedding viz (~4 entries).
- `demos_advanced.rs`: tiny LM, ViT, multi-head attention
  (~3 entries, mostly heavy).
- `demos_mlx.rs`: MLX-routed variants (~3 entries).

Each entry becomes `pub const NAME: Demo = Demo { ... };`.
`demos.rs` keeps the `Demo` struct + `ProgressNote` machinery
+ `pub const DEMOS: &[Demo] = &[crate::demos_basics::*, ...]`
facade.

Target: demos.rs 1179 -> ~150 lines (retires File-LOC FAIL).
Each topical sibling 150-250 lines (PASS).

### Step 002 -- split env.rs phase 1: extract var + param methods

Move the var / param method cluster from `impl Environment`
in `env.rs` to a new sibling `env_vars.rs`. Methods to move
(~10): `get`, `set`, `set_param`, `mark_param`, `is_param`,
`mark_frozen`, `unmark_frozen`, `is_frozen`, `params`,
`vars_iter`.

Rust allows multiple `impl Environment` blocks across
modules in the same crate. The struct + its fields stay in
`env.rs`; the sibling file just adds methods.

After this step env.rs drops ~10 methods (55 -> ~45). Still
FAIL but progress.

### Step 003 -- split env.rs phase 2: models + tokenizers + dirs

Move the model / tokenizer / data-dir / experiment-log
methods to `env_models.rs` and `env_dirs.rs`:

- `env_models.rs`: `get_model`, `models_iter`,
  `set_tokenizer`, `get_tokenizer`, `tokenizers_iter` (5
  methods).
- `env_dirs.rs`: `set_data_dir`, `data_dir`, `set_exp_dir`,
  `exp_dir`, `push_experiment_log`, `experiment_log` (6
  methods).

env.rs drops ~11 more methods (55 -> ~34).

### Step 004 -- split env.rs phase 3: device + peer

Move the device-stack + peer-dispatcher cluster to
`env_device.rs`:

- `device`, `push_device`, `pop_device`,
  `take_mlx_fallback_warning`, `tensor_device`,
  `set_tensor_device`, `set_peer_dispatcher`,
  `clear_peer_dispatcher`, `peer_dispatcher`,
  `set_device_tensor`, `get_device_tensor`,
  `remove_device_tensor` (12 methods, but on related state
  -- the device stack drives peer dispatch).

May need to split further into `env_device.rs` (stack +
fallback) + `env_peer.rs` (dispatcher + device-tensor) if
12 fns is over budget.

env.rs drops ~12 more methods (55 -> ~22).

### Step 005 -- split env.rs phase 4: tags + records + strings

Move per-value-type accessors to `env_values.rs`:

- `set_string`, `get_string`, `set_record`, `get_record`,
  `set_string_list`, `get_string_list`, `set_builtin_ref`,
  `get_builtin_ref`, `set_tag`, `get_tag`, `clear_tag`,
  `tags_iter` (12 methods).

May split into `env_tags.rs` + `env_values.rs` if needed.

env.rs drops ~12 more (55 -> ~10).

### Step 006 -- split env.rs phase 5: signals (metric + interrupt)

Move signal/lifecycle methods to `env_signals.rs`:

- `set_metric_sink`, `clear_metric_sink`, `metric_sink`,
  `emit_metrics`, `set_interrupt`, `clear_interrupt`,
  `check_interrupt` (7 methods).

env.rs drops ~7 more (55 -> ~3). Should now PASS the
fn-count budget.

### Step 007 -- model_dispatch.rs phase split (bonus)

If steps 001-006 hold + the saga has budget, attack
`model_dispatch.rs` (16 fns, 905 lines, 100-LOC
`apply_model`). The file conflates the constructor cluster
(`eval_linear`, `eval_attention`, etc.; start-up: build a
ModelSpec) with the dispatcher cluster (`apply_model`,
`apply_attention`, etc.; per-item: apply the spec).

Split into:

- `model_dispatch.rs`: the constructor cluster (~7
  `eval_*` builders for the model DSL).
- `model_apply.rs`: the dispatcher cluster (`apply_model`,
  `apply_attention`, etc.).

The 100-line `apply_model` itself needs the dataflow-
pipeline refactor: each match arm becomes a named helper
(`apply_linear`, `apply_chain`, etc.), `apply_model`
becomes a thin dispatch.

### Step 008 -- final ratchet + saga close

Final sw-checklist pass. Update `docs/language-status.md`
with the saga 33 close-out entry. Refresh `CHANGES.md`.
Mark the saga `complete --done`.
