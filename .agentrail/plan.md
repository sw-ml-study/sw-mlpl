# Saga 22: Feasibility Checking + Resource Estimation (v0.15.0)

## Why this exists

MLPL programs today happily accept `train 10000 {...}`
against a `chain(embed(32000, 1024, 0), ...)` without
warning that the resulting run will never fit on the
user's laptop. When real open-weights models
(SmolLM-135M, Llama-3.2-1B, Qwen2.5-0.5B) enter the
conversation the doom-loop possibilities multiply:
OOM on the GPU, out of disk on checkpoint save, or
wall-clock of days when the user expected minutes.

Saga 22 adds a **feasibility / resource estimator** so
the user can sanity-check a planned operation before
committing to it. The estimator is pure math over a
`ModelSpec` + dataset shape + train-loop parameters
(steps, batch size, sequence length) + device
characteristics. It returns param count, VRAM bytes
(forward + gradient + Adam moments + activation
heuristic), disk bytes per checkpoint, total FLOPS,
and wall-clock seconds. Users with limited hardware
can gate their actual train call behind a feasibility
check and avoid doomed runs.

Critically, the estimator does NOT require the actual
weights to work. A structural `ModelSpec` is enough;
that means the user can ask "how big would a SmolLM-
135M + LoRA(rank=8) run be on my laptop?" WITHOUT
needing a HuggingFace download path first. The actual
weights-loading story (safetensors parsing, HF
download, f16/bf16 tensor support) is a separate
future saga; this one sits at the estimation layer.

## Non-goals (deferred)

- **HuggingFace Hub download / safetensors loading.**
  Separate saga. The estimator can still talk about
  hypothetical HF models via a hardcoded spec table
  (SmolLM-135M / 360M / 1.7B, Llama-3.2-1B,
  Qwen2.5-0.5B) without ever pulling weights.
- **f16 / bf16 tensor support.** MLPL is f64 today.
  The estimator can compute VRAM for hypothetical
  half-precision runs via a `dtype_bytes` argument,
  but won't actually run them until a future saga
  adds the dtype machinery.
- **Dynamic profiling / exact measurement.** The
  estimator is honest-approximate: ~2x accuracy is
  the goal. Activation memory is architecture-
  sensitive; throughput varies by op mix. Real
  measurement needs a profiler, which is a separate
  surface.
- **Distributed / multi-GPU.** Single-device only.
  Saga 17 (CUDA + distributed) is the home for
  multi-device estimation.
- **Arbitrary-operation estimation.** Estimator
  targets the training-loop case (matmul-dominated
  param updates), not arbitrary MLPL programs. A
  scalar `sort` + `svg` pipeline gets no useful
  output; the user should not expect one.
- **Automatic recovery / shrinking.** The estimator
  reports; it does not auto-shrink the user's batch
  size, change precision, or otherwise rewrite the
  program. Returning actionable numbers is enough.

## Quality requirements (every step)

Identical to Sagas 16 / 16.5; `docs/sw-checklist-
patterns.md` is the decomposition reference.
Design for budgets up front.

## Phase 1 -- estimator builtin (1 step)

### Step 001 -- `estimate_train(...) -> [params, vram, disk, flops, wall]`

1. **Signature**.
   - `estimate_train(model, steps, batch_size,
     seq_len)` -- returns a rank-1 `[5]` f64 array
     with `[params, vram_bytes, disk_bytes, flops,
     wall_seconds]` in that fixed order. Labels
     pinned in the contract; no keyword-arg
     sugar because MLPL doesn't have records.
   - `model` -- a bare ident bound in `env.models`
     or an expression evaluating to
     `Value::Model`. Same argument shape as
     `freeze` / `lora` / `embed_table`.
   - `steps`, `batch_size`, `seq_len` -- positive
     scalar integers.

2. **Derivations**.
   - **Params:** walk `ModelSpec` (already
     implemented as `ModelSpec::params()`);
     sum each tracked parameter's shape product.
     Frozen params (LoRA base) still count for
     inference/forward memory but not for
     gradient/optimizer memory.
   - **VRAM bytes:** for each param P with size
     `|P|`:
     - `fwd: |P| * dtype_bytes` (the weight
       itself).
     - `grad: |P_trainable| * dtype_bytes` (zero
       for frozen).
     - `adam: 2 * |P_trainable| * dtype_bytes`
       (m + v moments).
     - Sum, plus an activation estimate:
       `activation = batch * seq * hidden * depth
       * dtype_bytes * activation_factor` where
       `hidden` is the max `d_model` seen in the
       spec, `depth` is the number of Attention
       + Linear + Embedding nodes, and
       `activation_factor = 4` is a conservative
       safety multiplier (attention scores + Q /
       K / V projections + residuals + softmax).
     - Default `dtype_bytes = 8` (f64, what
       MLPL uses); expose `dtype_bytes` as an
       optional 5th arg for what-if f32/f16 runs.
   - **Disk bytes:** `|P_all| * dtype_bytes` (one
     full checkpoint). A future addition can take
     a `save_every` argument and multiply.
   - **FLOPS:** per step, sum `2 * in * out *
     batch` for each Linear + LinearLora;
     `2 * seq^2 * d_model * batch` for each
     Attention node; `2 * batch * vocab *
     d_model` for each Embedding. Multiply by
     steps.
   - **Wall-clock:** `flops / device_throughput`.
     `device_throughput` defaults to a
     conservative `50e9` FLOPS (50 GFLOPS -- a
     CPU laptop lower bound). Override by setting
     `env.set_string("mlpl_device_throughput_
     gflops", ...)` (string for CLI env-var
     integration); step 002 ships a
     `calibrate_device()` builtin that writes this
     automatically.

3. **Module**: new
   `crates/mlpl-runtime/src/estimate_builtins.rs`.
   Design for budgets: 6 functions (orchestrator,
   param walk, VRAM, FLOPS, wall, try_call).
   Alternative -- put the model walk in
   `crates/mlpl-eval/src/` since it needs access
   to the ModelSpec tree. `mlpl-eval` already has
   `model_params` helpers; put a sibling there if
   the runtime side gets too thin.

4. **Contract**: new
   `contracts/eval-contract/estimate.md` --
   signature, per-component math, assumptions
   (f64 default, activation_factor = 4, flops
   model ignores softmax + layer norm + elementwise
   costs), non-goals (no dynamic profiling, no
   distributed, no f16 tensor support yet).

5. **TDD** in
   `crates/mlpl-eval/tests/estimate_tests.rs`:
   - **Tiny linear.** `m = linear(3, 4, 0);
     estimate_train(m, 100, 32, 1)` returns
     `[16, 16*8*4, 16*8, flops, wall]` --
     assert param count first, then the
     memory math, then flops = 2*3*4*32*100.
   - **Chain.** Two linears in a chain; params
     add, FLOPS add.
   - **LoRA.** `base = linear(10, 10, 0);
     student = lora(base, 4, 16.0, 7)` --
     base is frozen (W, b), adapters (A, B)
     are trainable; VRAM grad+adam terms only
     count adapters; params total includes
     both.
   - **Embedding.** `embed(100, 16, 0)` --
     params 100*16; flops model adds
     `2 * batch * 100 * 16` per step.
   - **Activation heuristic.** Attention in
     chain bumps activation memory; regression
     test that a chain with Attention gets more
     activation bytes than one without.
   - **dtype argument.** Pass dtype_bytes=4 via
     a 5-arg call; VRAM halves vs the 4-arg
     default.
   - **Error cases.** Non-model argument;
     negative steps / batch / seq; non-scalar
     args.
   - **Non-goals assertion.** Calling
     `estimate_train` on an
     `Activation(Tanh)`-only "model" (no
     trainable params) errors with "model has
     no trainable parameters".

6. Wire into dispatch via try_call chain. Returns
   `Value::Array` (rank-1 5 f64s).

## Phase 2 -- device calibration + what-if table + demo (1 step)

### Step 002 -- `calibrate_device()` + hypothetical models + demo

1. **`calibrate_device() -> gflops`** builtin.
   Runs a canned 1024x1024 matmul 10 times through
   `device::dispatched_call(env, "matmul", ...)`,
   measures wall-clock, computes observed GFLOPS,
   writes the result into `env.set_string("mlpl_
   device_throughput_gflops", "<value>")`. Returns
   the measured GFLOPS as a scalar. Device-aware:
   `device("mlx") { calibrate_device() }` writes
   a separate `mlpl_device_throughput_gflops_mlx`
   key so subsequent `device("mlx") {
   estimate_train(...) }` uses the measured MLX
   number.

2. **Hypothetical model spec table** -- a new
   builtin `hypothetical_model(name) -> ModelSpec`
   that returns a ModelSpec shaped like one of:
   - `"smollm-135m"` -- d_model=576, layers=30,
     heads=9, vocab=49152, seq=2048
   - `"smollm-360m"` -- d_model=960, layers=32,
     heads=15, vocab=49152, seq=2048
   - `"smollm-1.7b"` -- d_model=2048, layers=24,
     heads=32, vocab=49152, seq=2048
   - `"llama-3.2-1b"` -- d_model=2048, layers=16,
     heads=32, vocab=128256, seq=8192
   - `"qwen-2.5-0.5b"` -- d_model=896, layers=24,
     heads=14, vocab=151936, seq=2048
   Table hardcoded in the runtime crate. Returns
   a `ModelSpec::Chain` with `Embedding +
   N * Residual(Attention) + Linear head`
   structure -- enough for the estimator to walk,
   NOT enough to actually run (the param tables
   aren't populated in env). Errors helpfully if
   the caller tries `apply(hypothetical, X)`.

3. **`feasible(estimate_result, budget) -> 0/1`**
   convenience builtin. `budget` is a rank-1 `[3]`
   `[vram_bytes, disk_bytes, wall_seconds]`; all
   three must be satisfied (0 values in budget
   mean "don't check this dimension"). Returns
   scalar 1 if feasible, 0 otherwise. Pairs
   naturally with `if feasible(...) { train
   ... }` guards.

4. **`demos/feasibility.mlpl`**. Tour:
   - Estimate a mlpl-toy model (`chain(embed(32, 8,
     0), ...)`; `steps=100, batch=4, seq=8`).
     Print the estimate.
   - Estimate the same on MLX after a
     `calibrate_device()` call -- wall-clock
     drops.
   - Estimate a SmolLM-135M + `lora(rank=8)` fine-
     tune -- print VRAM, disk, wall. Compare
     against a budget.
   - Show `feasible(...)` gating a real training
     call on the mlpl-toy model.
   - CLI-only (browser can't measure CPU ops for
     calibration meaningfully; skip on WASM).

5. **`docs/using-feasibility.md`** retrospective +
   user guide. Sections:
   - Status block (shipped Saga 22 / v0.15.0).
   - What the estimator computes and what it
     does NOT (the ~2x accuracy target, the
     activation_factor heuristic).
   - Signature + usage patterns.
   - Hypothetical-model table with the specs
     and sources.
   - The LoRA-on-SmolLM what-if worked example.
   - `feasible(...)` guard pattern.
   - Non-goals / deferred list.

6. **`docs/configurations.md`**. Add rows for
   `estimate_train`, `calibrate_device`,
   `hypothetical_model`, `feasible` in the
   CLI-vs-web matrix. Estimator itself works in
   web; calibration needs CLI (browser timing is
   too noisy + device is always pure-WASM).

## Phase 3 -- release (1 step)

### Step 003 -- release v0.15.0

1. Bump `Cargo.toml` workspace.package.version
   `0.14.1 -> 0.15.0`. Minor-level bump (new
   language surface).
2. `CHANGELOG.md` new v0.15.0 section above
   v0.14.1: documents `estimate_train`,
   `calibrate_device`, `hypothetical_model`,
   `feasible`, the demo, the docs.
3. `docs/saga.md` Saga 22 retrospective above
   Saga 16.5.
4. `docs/status.md` Saga 22 moves from Planned
   (insert it!) to Completed. Re-order "Next saga"
   pointer to Saga 19 (LLM-as-tool REST) next.
5. `cargo build --release`.
6. `./scripts/build-pages.sh` and commit pages/.
7. `./scripts/gen-changes.sh` and commit
   CHANGES.md.
8. `/mw-cp` quality gates.
9. Tag `v0.15.0` locally; DO NOT push without
   explicit user confirmation.
10. `agentrail complete --done`.

## Dependency graph

```
001 estimate_train builtin
        |
002 calibrate + hypothetical + feasible + demo
        |
003 release v0.15.0
```

Sequential; each step depends on the previous.

## After Saga 22

Saga 19 (LLM-as-tool REST, `llm_call` builtin) is
next in the sequence -- its 3-step plan survives in
`.agentrail-archive/mlpl-llm-rest-20260424T095011/`
and can be re-initialized verbatim.
