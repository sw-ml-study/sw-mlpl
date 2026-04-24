Phase 1 step 001: `estimate_train(...) -> [5]`
feasibility / resource-estimation builtin.

Users on limited hardware need a way to sanity-check a
planned train call BEFORE committing to it -- OOM, out
of disk, hours-of-compute surprises all preventable
with a pure-math estimate over the ModelSpec + loop
shape. This step ships the core estimator; step 002
adds device calibration, a hypothetical-model table
for SmolLM-scale what-ifs, and a `feasible(...)` guard.

1. **Signature**.
   - `estimate_train(model, steps, batch_size,
     seq_len)` -- required 4-arg form; default
     `dtype_bytes = 8` (f64, what MLPL uses).
   - `estimate_train(model, steps, batch_size,
     seq_len, dtype_bytes)` -- optional 5-arg
     form, pass `4` for f32, `2` for f16/bf16
     what-ifs.
   - Returns rank-1 `[5]` f64:
     `[params, vram_bytes, disk_bytes, flops,
     wall_seconds]` in that fixed order.
   - `model` -- bare ident bound in `env.models`
     or expression evaluating to `Value::Model`.
   - `steps`, `batch_size`, `seq_len` -- positive
     scalar integers (non-integer or non-positive
     values error).

2. **Derivations** (all documented in the
   contract):
   - **params = sum(|P|) for P in model.params()**
     (use existing `ModelSpec::params()` +
     `env.get(name).shape()`).
   - **trainable = sum(|P|) for P in
     model.params() where !env.is_frozen(P)**.
   - **vram_bytes =** `(params + trainable +
     2*trainable) * dtype_bytes + activation`
     where `activation = batch * seq * hidden *
     depth * dtype_bytes * 4`. `hidden` is the
     max `d_model` seen across
     `Embedding { d_model, .. }`, `Attention {
     d_model, .. }`, and `LinearLora { out_dim,
     ..}`-style nodes, defaulting to the widest
     Linear's `out` dim if no d_model is
     present. `depth` is the count of
     parameterized nodes (Linear, LinearLora,
     Embedding, Attention). `4` is the
     activation safety factor (activations +
     attention scores + residuals + softmax
     temporaries).
   - **disk_bytes = params * dtype_bytes** (one
     full checkpoint; save-frequency multiplier
     is deferred).
   - **flops per step:**
     - `Linear / LinearLora: 2 * in_dim * out_dim
       * batch`
     - `Embedding: 2 * batch * vocab * d_model`
     - `Attention: 2 * seq^2 * d_model * batch`
       (QK^T + softmax-times-V scaling). Does
       NOT include QKV projections (those are
       counted as their parent Linear nodes
       inside the Attention's inner structure;
       the estimator walks those).
     - Activation / RmsNorm / Residual wrapper
       ops are ignored (sub-dominant).
   - **flops total = flops_per_step * steps**.
   - **wall_seconds = flops / throughput**, where
     throughput is fetched from
     `env.get_string("mlpl_device_throughput_
     gflops")` parsed as f64, defaulted to
     `50.0` if unset (conservative
     laptop-CPU lower bound). Multiply by 1e9
     to convert to FLOPS.

3. **Module**: new
   `crates/mlpl-eval/src/model_estimate.rs`
   (lives in eval, not runtime, because it needs
   Environment access to read param shapes +
   frozen set). Budget-conscious from the start:
   6 functions (`eval_estimate_train`
   orchestrator, `resolve_model_spec`, `sum_params`,
   `sum_trainable`, `compute_vram`,
   `compute_flops`). Keep `compute_wall_seconds`
   inlined into orchestrator to stay at 6.

4. Wire as a new FnCall branch in
   `crates/mlpl-eval/src/eval.rs` alongside the
   `embed_table` dispatch site; returns
   `Value::Array`.

5. **Contract**: new
   `contracts/eval-contract/estimate.md` --
   signature, math for every derived field,
   assumptions (f64 default; `activation_factor =
   4`; FLOPS ignores softmax / layer norm /
   elementwise / residual wrappers; throughput
   is a user-settable env key), 2x accuracy
   target, error cases, non-goals (no dynamic
   profiling, no distributed, no f16 tensor
   support yet, no auto-shrinking).

6. **TDD** (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-eval/tests/estimate_tests.rs`:
   - **Tiny linear.**
     `m = linear(3, 4, 0); estimate_train(m, 100,
     32, 1)`:
     - params = 3*4 + 1*4 = 16
     - trainable = 16 (no freeze)
     - vram = (16 + 16 + 32)*8 + 32*1*4*1*8*4
       = 512 + 4096 = 4608
       (depth=1, hidden=4)
     - disk = 16*8 = 128
     - flops = 2*3*4*32*100 = 76800
   - **LoRA freeze.** `base = linear(10, 10, 0);
     student = lora(base, 4, 16.0, 7);
     estimate_train(student, ...)`:
     - params includes W, b, A, B
     - trainable = |A|+|B|; base frozen
     - VRAM grad+adam terms drop accordingly
     - flops includes only the adapter matmuls
       that train (but forward flops still
       include the base) -- OR flops include
       the full forward path regardless of
       freeze since the forward pass cost is
       the same. Go with: flops count every
       matmul once per step (forward); freeze
       only affects VRAM.
   - **Embedding vocab scaling.** V=1000 vs
     V=100 -- params and flops scale linearly.
   - **Chain additivity.** A chain of two
     linears has params + flops = sum of each.
   - **dtype_bytes override.** Same model, dtype=4:
     vram halves; disk halves.
   - **Throughput override.** Set
     `env.set_string("mlpl_device_throughput_
     gflops", "500.0")` before the call; wall
     seconds should drop 10x relative to the
     default `50.0`.
   - **Error cases.** Non-model first arg;
     negative steps / batch / seq; non-integer
     scalar; rank-0 model (activation-only chain).
   - **Non-trainable model.** A chain that is
     only Activations -- error "model has no
     trainable parameters".

7. Quality gates + commit. Commit message
   references Saga 22 step 001.
