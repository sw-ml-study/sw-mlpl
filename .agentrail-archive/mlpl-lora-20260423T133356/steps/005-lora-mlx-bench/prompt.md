Phase 2 step 005: LoRA fine-tune on MLX + Criterion bench.

1. New `demos/lora_finetune_mlx.mlpl`. Identical to
   `demos/lora_finetune.mlpl` but with the fine-tune
   training loop (the adam calls on `student`) wrapped in
   `device("mlx") { ... }` with the standard
   `to_device(student, "mlx")` + `to_device(instr_X,
   "mlx")` prologue. Base pre-training stays on CPU
   (cheap + deterministic; mirrors the Saga 20 pattern).

2. Integration test
   `crates/mlpl-eval/tests/lora_mlx_demo_tests.rs`,
   triple-gated on macOS + aarch64 + `mlx` feature.
   - Run both CPU and MLX cut-down paths with identical
     seeds.
   - Assert fine-tune loss curves agree elementwise
     within fp32 tolerance (1e-3) across all training
     steps.
   - Assert adapter values (A, B) after the final step
     agree within fp32 tolerance.
   - Assert frozen base values stay bit-identical on
     BOTH paths (the frozen-param rule must be
     backend-independent).
   - Belt-and-braces: the `demos/lora_finetune_mlx.mlpl`
     file itself parses.

3. Add a `lora_finetune_step` Criterion group to
   `crates/mlpl-bench/benches/mlx_vs_cpu.rs`. Each
   iteration: build a small lora-wrapped model, freeze
   the base, run one adam step. Cold timings +
   warm timings as usual. Sample size relaxed to 20 (the
   step is chunky).

4. Run the bench manually once; capture CPU vs MLX warm
   numbers and record them in `docs/benchmarks.md`
   alongside the existing neural_thicket_variant_loop
   and tiny_lm_train_step rows. Be honest about the
   ratio; if it's below 1.0x (MLX slower) that's
   consistent with Saga 14 / Saga 20 at this scale and
   should be documented as expected. If the ratio is
   notably different (> 0.5x gap from Saga 20), add a
   one-paragraph note explaining why.

5. If LoRA + MLX exposes any backend gap (e.g. the extra
   matmul chain triggers a tape gotcha, or the frozen-
   param filter misbehaves under MLX dispatch), open a
   targeted follow-up issue rather than expanding scope
   into step 005.

6. Quality gates + `/mw-cp`. Commit message references
   Saga 15 step 005 and includes the measured warm ratio.
