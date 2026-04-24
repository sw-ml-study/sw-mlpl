Phase 2 step 002: `calibrate_device()` +
`hypothetical_model(name)` + `feasible(est, budget)` +
`demos/feasibility.mlpl` + docs.

With `estimate_train` landed in step 001, now make the
wall-clock honest (calibrate the device once per
session), add a way to ask about hypothetical HF-scale
models WITHOUT needing the weights locally, and ship
the compose-a-guard pattern.

1. **`calibrate_device() -> gflops`** builtin.
   - Zero-arg; runs a canned benchmark: 10
     iterations of a 1024x1024 matmul via
     `device::dispatched_call(env, "matmul",
     ...)` on freshly-randn'd inputs.
   - Measures wall-clock around the 10-iter
     loop (first iter is a warmup and discarded
     from the average).
   - Computes observed throughput:
     `2 * 1024^3 * 9 / elapsed_seconds` = FLOPS;
     divide by 1e9 for GFLOPS.
   - Writes the result to
     `env.set_string("mlpl_device_throughput_
     gflops", format!("{gflops}"))`.
   - Device-aware: if the current active device
     is `"mlx"`, writes
     `mlpl_device_throughput_gflops_mlx` instead;
     `estimate_train`'s lookup is
     device-aware in the same way.
   - Returns the measured GFLOPS as a scalar.

2. **`hypothetical_model(name) -> Value::Model`**
   builtin. Hardcoded spec table:
   - `"smollm-135m"`: d_model=576, layers=30,
     heads=9, vocab=49152, intermediate=1536
   - `"smollm-360m"`: d_model=960, layers=32,
     heads=15, vocab=49152, intermediate=2560
   - `"smollm-1.7b"`: d_model=2048, layers=24,
     heads=32, vocab=49152, intermediate=8192
   - `"llama-3.2-1b"`: d_model=2048, layers=16,
     heads=32, vocab=128256, intermediate=8192
   - `"qwen-2.5-0.5b"`: d_model=896, layers=24,
     heads=14, vocab=151936, intermediate=4864
   Returns a `ModelSpec::Chain` containing:
   - `Embedding { vocab, d_model, table: "hyp_
     emb" }`
   - `layers * Residual(Chain(RmsNorm, Attention
     { d_model, heads, causal=true, ... },
     Linear { intermediate -> d_model }))`
   - `Linear { d_model -> vocab }` head
   **Does not populate `env`** with the underlying
   parameter arrays. Any subsequent
   `apply(hypothetical, X)` errors helpfully
   with "hypothetical_model spec has no materialized
   weights; use estimate_train for resource
   estimation only".

3. **`feasible(estimate_result, budget) -> 0/1`**
   builtin.
   - `estimate_result` -- rank-1 `[5]` f64 from
     `estimate_train`.
   - `budget` -- rank-1 `[3]` f64
     `[vram_budget_bytes, disk_budget_bytes,
     wall_budget_seconds]`. A zero in any slot
     means "no budget constraint for this
     dimension" (skip the check).
   - Returns 1.0 if every non-zero budget is
     satisfied (estimate <= budget); 0.0
     otherwise.
   - Wrong arity, wrong shape -> error.

4. **Module**: extend
   `crates/mlpl-runtime/src/estimate_builtins.rs`
   (or add a sibling
   `crates/mlpl-eval/src/model_estimate_helpers.rs`
   if budget pressure makes it cleaner). The
   calibrate_device implementation MUST live
   wherever `device::dispatched_call` is callable;
   that's eval, not runtime. Put
   `feasible` in runtime (pure array math) and
   `calibrate_device` + `hypothetical_model` in
   eval.

5. **Tests**.
   - `crates/mlpl-eval/tests/estimate_calibrate_
     tests.rs`: `calibrate_device()` returns a
     positive number on CPU; writes the
     expected string key. Skip / gate the test
     if MLX isn't available; do NOT measure
     MLX in CI because the env is not
     deterministic.
   - `crates/mlpl-eval/tests/hypothetical_model_
     tests.rs`: `hypothetical_model("smollm-
     135m")` returns a Model whose
     `params()` walks without panic; param
     count matches an expected range (135M +/-
     10% since our structure is approximate);
     `apply(m, X)` errors with the expected
     message.
   - `crates/mlpl-runtime/tests/feasible_tests.rs`
     (or inline in eval tests): feasible with a
     small estimate and a big budget -> 1;
     feasible with a small estimate and a tiny
     vram budget -> 0; feasible with a zero
     budget skip -> 1.

6. **Wire into dispatch**. `calibrate_device` and
   `hypothetical_model` return respectively an
   array scalar and a Value::Model -- match the
   existing patterns (`iota` for array-return,
   `linear` / `embed` for model-return).
   `feasible` returns an array scalar.

7. **Contracts**:
   - `contracts/eval-contract/calibrate-device.md`
     -- what gets measured, how the result is
     cached, device-awareness, accuracy caveats
     (throughput varies 2x+ with matmul
     dimensions).
   - `contracts/eval-contract/hypothetical-
     model.md` -- the spec table, the
     "estimation-only" contract, the
     apply-errors-helpfully rule.
   - `contracts/eval-contract/feasible.md` --
     signature, zero-means-skip semantics,
     error cases.

8. **`demos/feasibility.mlpl`** CLI-only demo:
   - Small mlpl-toy model; call
     `estimate_train(...)`; print result.
   - `calibrate_device()`; re-estimate;
     wall-clock drops.
   - `hypothetical_model("smollm-135m")`;
     `lora(base, 8, 16.0, 0)` if possible (it
     isn't -- hypothetical models are
     estimation-only, so wrap the lora call in
     a structural rewrite that returns a
     LoRA'd spec without actually initializing
     adapter tables -- OR document that the
     user should estimate the base + add the
     adapter arithmetic by hand for now. Pick
     the simpler path during implementation).
   - `feasible(est, [4_000_000_000,
     10_000_000_000, 600.0])`: VRAM budget 4GB,
     disk 10GB, wall 10 min. Gate a subsequent
     train call on the result.

9. **`docs/using-feasibility.md`** retrospective.

10. Update the `docs/configurations.md` CLI-vs-web
    matrix with the four new builtins.

11. Quality gates + commit. Commit message
    references Saga 22 step 002.
