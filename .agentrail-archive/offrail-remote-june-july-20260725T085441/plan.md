# Saga: offrail-remote-june-july (retroactive)

Retroactive record of development done 2026-06-12 through 2026-07-21 on a
remote host without agentrail. Reconstructed on 2026-07-25 from first-parent
git history (range 36a45e6b..8728fb22, 123 commits) via `agentrail audit`.

Work groups, in landing order:

1. gpu-workspace-split -- 4-stage split of CUDA/MLX compute into sibling
   crates (mlpl-cuda-eval, mlpl-mlx-eval), device-trait seam, three build
   modes, post-migration fixups.
2. macos-telemetry-sparklines -- macOS /v1/stats backend (no sudo), live
   sparkline fixes, Apple Silicon connect-server runbook.
3. connect-ask-hardening -- :ask grounding, model selection, MLPL reference
   prompt, verbatim prompts, demo-line routing, HTTPS/HTTP honesty.
4. introspection-ux-viz3d -- :list <fn>, indented UDF source in 3D view.
5. beginner-ml-comprehension -- plan doc + train_val_curve, loss_landscape,
   gradient-flow demos, weight-decay/LR demos, beginner spine paths.
6. content-toml-codegen -- lessons/demos/paths content moved to TOML +
   build.rs codegen.
7. quant-glossary-search -- quantization teaching path, glossary quant
   terms, fuzzy search + glossary crate splits.
8. spec-decoding-plan -- speculative decoding (MLX) plan docs.
9. apl2-stage1-introspection -- depth/disp/size/tally builtins + docs.
10. changelog-pages-chores -- CHANGES.md refreshes + pages rebuilds.
