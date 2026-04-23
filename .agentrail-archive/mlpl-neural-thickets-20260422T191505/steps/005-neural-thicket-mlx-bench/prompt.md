Phase 2 step 005: MLX variant loop + `mlpl-bench` entry.

Prove the speedup end-to-end: the variant-loop portion of the
Neural Thickets demo on MLX, with honest Criterion numbers.

1. Wrap the perturb/score portion of
   `demos/neural_thicket.mlpl` in `device("mlx") { ... }` with
   a `to_device(base, "mlx")` prologue, mirroring the
   Saga 14 pattern from `demos/tiny_lm_mlx_demo.mlpl`. Base
   training stays on CPU (cheap + deterministic); the
   variant-loop is where the MLX thesis applies.
2. Integration test
   `crates/mlpl-eval/tests/neural_thicket_mlx_demo_tests.rs`
   under the existing MLX cfg gate
   (`#[cfg(all(target_os = "macos", target_arch = "aarch64",
   feature = "mlx"))]` or equivalent) that runs the MLX
   cut-down and asserts:
   - Heatmap shape equals the CPU path's.
   - All losses finite, agree with CPU numerics within
     documented fp32 tolerance.
   - `argtop_k` + ensemble shape invariants match.
3. Add a `neural_thicket_mlx` Criterion entry to
   `crates/mlpl-bench` that runs the MLX variant-loop on a
   fixed cut-down. Capture numbers; record them in
   `docs/benchmarks.md` alongside the existing Tiny LM MLX
   row. Report honestly -- Saga 14 already established that
   MLX can be slower than CPU on small workloads; if the
   variant loop is similarly small, say so in the doc.
4. If `device("mlx")` inside a `repeat N { }` exposes any
   MLX gap (tape rematerialization, param-name hashing under
   `clone_model`, etc.), open a targeted follow-up -- do NOT
   expand scope into step 005.
5. Quality gates + `/mw-cp`. Commit message references
   Saga 20 step 005 and lists the measured MLX-vs-CPU ratio.
