Phase 3 step 003: release v0.15.0.

1. **Version bump**. `Cargo.toml`
   workspace.package.version `0.14.1 -> 0.15.0`.
   Minor bump because the surface adds four new
   language-level builtins.

2. **`CHANGELOG.md`** v0.15.0 section above
   v0.14.1. Added items:
   - `estimate_train(model, steps, batch_size,
     seq_len [, dtype_bytes]) -> [5]` --
     pure-math feasibility estimator: param
     count, VRAM bytes, disk bytes per
     checkpoint, total FLOPS, wall-clock
     seconds. Targets ~2x accuracy. Contract:
     `contracts/eval-contract/estimate.md`.
   - `calibrate_device() -> gflops` -- one-shot
     matmul benchmark on the active device;
     caches result in `env.set_string(...)` so
     subsequent `estimate_train` calls read
     honest wall-clock numbers. Device-aware
     (CPU / MLX separate cache keys). Contract:
     `contracts/eval-contract/calibrate-
     device.md`.
   - `hypothetical_model(name) ->
     Value::Model` -- structural ModelSpec for
     SmolLM-135M / 360M / 1.7B, Llama-3.2-1B,
     Qwen2.5-0.5B. Works with `estimate_train`
     for what-if sizing; errors on `apply(...)`
     (no materialized weights). Contract:
     `contracts/eval-contract/hypothetical-
     model.md`.
   - `feasible(estimate_result, budget) -> 0/1`
     -- guard pattern: `if feasible(est, [vram,
     disk, wall]) { train ... }`. Zeroes in
     budget mean "skip this check". Contract:
     `contracts/eval-contract/feasible.md`.
   Scope notes: activation memory is a 4x
   safety-factor heuristic; FLOPS model ignores
   softmax / layer norm / elementwise / residual
   wrappers; f64 default (f16/bf16 are what-if
   only until a future saga adds the tensor
   machinery); HuggingFace download +
   safetensors loading are a separate future
   saga; distributed / multi-GPU estimation is
   Saga 17 territory.

3. **`docs/saga.md`** -- Saga 22 retrospective
   above Saga 16.5.

4. **`docs/status.md`** -- Saga 22 row moves from
   Planned into Completed. Update the "Next saga"
   pointer to Saga 19 (LLM-as-tool REST) -- its
   plan + 3 steps are preserved in
   `.agentrail-archive/mlpl-llm-rest-20260424T
   095011/` and can be re-initialized verbatim.

5. `cargo build --release`.

6. `./scripts/build-pages.sh` and commit pages/
   (new builtins surface on any web REPL lesson
   intro that mentions them).

7. `./scripts/gen-changes.sh` and commit the
   refreshed CHANGES.md.

8. `/mw-cp` quality gates (cargo test, clippy,
   fmt, markdown-checker on changed files, sw-
   checklist baseline held).

9. Tag `v0.15.0` locally. DO NOT push the tag
   without explicit user confirmation (v0.12.0 /
   v0.13.0 / v0.14.0 / v0.14.1 cadence).

10. `agentrail complete --done`.
