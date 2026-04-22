Phase 5 step 010: release v0.11.0.

Cut the MLX release.

1. Bump workspace version to `0.11.0` (follow the pattern
   used by the Saga 13 v0.10.0 release commit). All crate
   versions bump in lockstep.
2. Release commit message summarizes what shipped:
   - `mlpl-mlx` runtime target behind the `mlx` feature
   - `device("...") { body }` scoped form in the language
   - `to_device(x, "mlx" / "cpu")` movement helpers
   - Autograd + optimizers + `train { }` on MLX
   - Tiny LM MLX variant (`demos/tiny_lm_mlx.mlpl`) with
     measured speedup (quote the step 008 number)
   - Benchmarks on CPU vs MLX for the Criterion workloads
   - "Running on MLX" tutorial lesson
3. Tag `v0.11.0`. Push commit and tag.
4. Verify the pages workflow deploys the updated demo and
   tutorial list: `gh run list --workflow=pages.yml
   --limit 1` and confirm the deployed site shows the new
   lesson.
5. Saga 14 is DONE -- `agentrail complete` uses `--done`.
6. Quality gates + `/mw-cp`. Commit message references step 010
   and v0.11.0.
