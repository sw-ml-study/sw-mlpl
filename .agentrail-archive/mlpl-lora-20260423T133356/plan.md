# LoRA Fine-Tuning Milestone (Saga 15, v0.13.0)

## Why this exists

Saga 13 showed the platform thesis on a from-scratch Tiny LM.
Saga 14 put that same forward pass on MLX. Saga 20 showed a
weight-perturbation research workflow. The next natural move
is the other direction: take a trained base, freeze its
weights, and add a small set of trainable low-rank adapters
that specialize the base on a new task. That is LoRA (Low-
Rank Adaptation) -- the dominant fine-tuning technique in the
open-weight LLM ecosystem.

Saga 15 adds LoRA as a first-class MLPL operation: a `lora(m,
rank, alpha, seed)` builtin that walks a model's
`ModelSpec` tree, replaces every `Linear` node with a
lora-wrapped variant that owns two low-rank adapter matrices
`A: [in, r]` and `B: [r, out]` alongside the frozen base
`W: [in, out]`. Forward becomes
`y = X @ W + (alpha / r) * X @ A @ B + b`; the base `W`, `b`
stay frozen and only `A`, `B` train. A small
instruction-tuning demo shows the base + adapter composing
into a specialized model in ~40 extra lines of MLPL source.

## Non-goals (deferred)

- **QLoRA / 4-bit quantization.** Quantization needs careful
  per-tensor scale/zero-point handling and its own parity
  harness; quantization lands in its own saga once LoRA
  numerics are proven. Saga 15 trains the base in fp64/fp32
  as today.
- **Layer-scoped adapter subsets.** LoRA in the literature
  often only adapts attention projections (Wq, Wv), not MLP
  linears. Saga 15 ships the uniform "every Linear gets an
  adapter" variant; selective attachment is a follow-up
  (plus it composes with Saga 20's family walker).
- **Adapter merging.** `merge_lora(m)` that bakes the
  adapter back into the frozen W is a nice-to-have for
  inference deployment; deferred until an inference path
  actually cares.
- **Multi-adapter composition / adapter routing.** Shared
  subspaces, adapter stacking, and routing are the
  follow-up saga after plain LoRA proves out.
- **Real LLM checkpoints.** The demo uses a Saga 13 Tiny LM
  base. A Llama-class base needs Saga 15+ checkpoint format
  or Saga 19's LLM sidecar; both are out of scope here.

## Quality requirements (every step)

Identical to Saga 20:

1. TDD: failing test first, then implementation, then refactor.
2. Quality gates must all pass before commit:
   - `cargo test`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo fmt --all` + `cargo fmt --all -- --check`
   - `markdown-checker -f "**/*.md"` if docs touched
   - `sw-checklist`
3. Use `/mw-cp` checkpoint process.
4. Push immediately after commit.
5. Web UI changes rebuild `pages/` via `scripts/build-pages.sh`.
6. `.agentrail/` changes are committed whenever they change.
7. New FAILs in `sw-checklist` must be resolved in the same
   step (extract modules / split functions as needed).

## What already exists

- Saga 13 Tiny LM (`demos/tiny_lm.mlpl`) with `embed`,
  `chain`, `residual`, `rms_norm`, `causal_attention`,
  `linear`, `relu_layer`, `cross_entropy`, `adam`, `train { }`.
- Saga 20 `clone_model`, `perturb_params`: param-walking
  infrastructure in `crates/mlpl-eval/src/model_clone.rs`
  and `model_perturb.rs` that the LoRA rewrite can reuse.
- `ModelSpec` enum with 7 variants (`crates/mlpl-eval/src/model.rs`).
  Forward `apply` + autograd tape in `model_dispatch.rs` +
  `model_tape.rs`.
- Adam / momentum_sgd optimizer state in `grad.rs`; tracks
  trainable params via `env.params: HashSet<String>`.
- MLX backend for forward + autograd (Saga 14); LoRA will
  compose with `device("mlx") { }` via the same dispatch
  the existing Model DSL uses.

## Phase 1 -- LoRA primitives (3 steps)

### Step 001 -- `freeze(m)` + `env.frozen_params`
Add a second param set to `Environment`: `frozen_params:
HashSet<String>` (alongside the existing `params`). A
parameter name can be in both `params` and `frozen_params`
-- it still has a value, but optimizers (`adam`,
`momentum_sgd`) must skip any name present in
`frozen_params`. `freeze(m)` walks a model's
`ModelSpec::params()` and adds every name to
`frozen_params`; returns the model unchanged (scalar zero
unit, per the `to_device` / `perturb_params` convention).
An inverse `unfreeze(m)` is a nice-to-have; decide during
TDD.

Contract: `contracts/eval-contract/freeze.md`. TDD:
`crates/mlpl-eval/tests/freeze_tests.rs`:
- After `freeze(m)`, every `m.params()` name is in
  `env.frozen_params` AND still in `env.params`.
- `adam` on a frozen model leaves all param values unchanged
  across N training steps.
- Partial freeze (freeze one model, train another) works.
- Round-trip: unfreeze + train moves params again.

### Step 002 -- `lora(m, rank, alpha, seed)` builtin
New `ModelSpec::LinearLora { w, b, a, b_adapter, in_dim,
out_dim, rank, alpha }` variant. `lora(m, r, alpha, seed)`
clones `m`'s tree (reusing the `model_clone` infrastructure)
and replaces every `Linear { w, b }` node with a
`LinearLora` that keeps the cloned `w`, `b` and allocates
two fresh adapter params:
- `a_name = format!("__lora_A_{id}")`, shape `[in_dim,
  rank]`, initialized as `randn(seed, [in_dim, rank]) * (1.0
  / sqrt(in_dim))` (small; standard LoRA init).
- `b_adapter_name = format!("__lora_B_{id}")`, shape
  `[rank, out_dim]`, initialized to zeros (so the
  pre-training-step delta is zero and `apply(lora_m, X)` ==
  `apply(m, X)` on entry).

The returned model's `params()` method contributes ALL
parameter names (frozen + adapter); `freeze` callers can
mark the frozen subset separately. Document the init
convention + the zero-init-on-B rationale in the contract.

Contract: `contracts/eval-contract/lora.md`. TDD:
`crates/mlpl-eval/tests/lora_tests.rs`:
- `lora(linear(4, 8, 0), 2, 4.0, 7)` returns a model with
  W [4, 8], b [1, 8], A [4, 2], B [2, 8].
- B is zero-initialized, so `apply(lora_m, X) == apply(m, X)`
  before any training.
- Clone-of-a-clone works; adapter names are disjoint across
  clones.
- `lora` applied to a chain walks every Linear and leaves
  non-linear nodes (`Residual`, `RmsNorm`, `Embedding`,
  `Attention`) unchanged.
- `rank <= 0` or `rank > min(in_dim, out_dim)` -- pick the
  reasonable error.

### Step 003 -- Forward + autograd for `LinearLora`
Extend `apply_model` in `model_dispatch.rs` to handle the new
variant: forward through a `LinearLora` computes
`X @ W + (alpha / rank) * X @ A @ B + b`. Extend the autograd
tape (`model_tape.rs`) so `grad(cross_entropy(apply(lora_m,
X), Y), lora_m)` produces gradients on A, B (and on W, b if
the caller has NOT called `freeze`). Frozen params receive
gradient-zero and `adam` skips them (step 001's invariant).

TDD: `crates/mlpl-eval/tests/lora_forward_tape_tests.rs`:
- Forward numerical check: hand-construct W, A, B, b; verify
  `apply(lora_m, X)` matches the manual formula.
- Gradient check: finite-difference against a small lora_m,
  confirm dL/dA and dL/dB match autograd within
  1e-4.
- Frozen W, b receive zero optimizer updates across 3 Adam
  steps.
- MLX parity: `device("mlx") { apply(lora_m, X) }` matches
  the CPU path within fp32 tolerance.

## Phase 2 -- Training + demo (2 steps)

### Step 004 -- End-to-end LoRA fine-tune on a tiny instruction corpus
New `demos/lora_finetune.mlpl`:
1. Load the Saga 13 Tiny LM body; train it briefly on
   Shakespeare.
2. Build a tiny synthetic instruction corpus (10-20 Q/A
   pairs encoded as a single string, e.g. `"Q: hello A:
   world\n Q: name A: alice\n..."`) and a held-out pair.
3. `student = lora(base, 8, 16.0, 0)`; `freeze(student)`
   does NOT freeze the adapters because `freeze` operates on
   the SOURCE model, not on the adapter-specific names --
   we freeze the base weights by calling
   `freeze(base)` BEFORE `lora(...)` if desired, OR we
   freeze only the non-adapter params after the LoRA rewrite.
   Pick during TDD; document in the contract.
4. Train `student` on the instruction corpus. Verify that
   the base params are unchanged and the adapter params
   move.
5. Render the loss curve; `experiment` captures the
   instruction-tuning run.

Integration test
`crates/mlpl-eval/tests/lora_finetune_tests.rs` runs a
cut-down variant (V=32, d=8, 5 base-train + 5 lora-finetune
steps) and asserts:
- Base W, b unchanged after LoRA training.
- Adapter A moved from its init; adapter B moved from zero.
- Loss on the instruction corpus decreases across training.

### Step 005 -- LoRA on MLX + bench
Wrap the instruction-tuning loop of `lora_finetune.mlpl` in
`device("mlx") { }`. Integration test
`lora_mlx_demo_tests.rs` asserts CPU-MLX parity on both the
forward pass and the optimizer step within fp32 tolerance.
Add a `lora_finetune_step` Criterion group to
`mlpl_vs_cpu.rs` and record the number in
`docs/benchmarks.md`.

## Phase 3 -- Tutorial, docs, release (2 steps)

### Step 006 -- Tutorial lesson + `docs/using-lora.md`
Add a "LoRA Fine-Tuning" web REPL tutorial lesson following
the Saga 20 "Neural Thickets" pattern: short narrative, a
tiny interactive example (LoRA-wrap a `linear(4, 8, ...)`,
take one training step, inspect the adapter), `svg(...,
"bar")` or similar viz of the adapter norms. Rebuild
`pages/` via `scripts/build-pages.sh`.

`docs/using-lora.md` covers: what LoRA is and why, the four
builtins (`freeze`, `unfreeze` if it exists, `lora`,
follow-up `merge_lora`), the zero-init-B rationale, the
`alpha / rank` scaling, the demo walkthrough, measured MLX
vs CPU numbers, and the deferred follow-up surface (QLoRA,
selective layer attachment, adapter merging, multi-adapter
routing).

### Step 007 -- Release v0.13.0
Bump workspace version 0.12.0 -> 0.13.0. Update
`CHANGELOG.md` with the v0.13.0 section. Mark Saga 15
complete in `docs/saga.md` (entry above Saga 20) and
`docs/status.md` (row move Planned -> Completed). Tag
v0.13.0 locally. Confirm before pushing the tag per
CLAUDE.md / project policy. `agentrail complete --done`
closes the saga.

## Dependency graph

```
001 freeze + frozen_params
  \-- 002 lora builtin + LinearLora variant
        \-- 003 forward + autograd for LinearLora
              \-- 004 lora_finetune CPU demo
                    \-- 005 lora_finetune MLX + bench
                          \-- 006 tutorial + using-lora.md
                                \-- 007 release v0.13.0
```

Steps are strictly sequential; each step's test suite uses
the prior steps' surface as a fixture.
