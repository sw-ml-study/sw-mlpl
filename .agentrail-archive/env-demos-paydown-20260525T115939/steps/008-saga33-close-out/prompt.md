Saga 33 step 008 (FINAL, use --done): saga close-out.

1. Final sw-checklist pass. Quote start-of-saga (143 fails / 450 warnings; this is saga 32's end) vs end-of-saga counts. Realistic target: -15 to -20 fails.

2. Verify env.rs is at <=7 methods (Module-Function-Count FAIL retired).
3. Verify demos.rs is at <500 lines (File-LOC FAIL retired).
4. Verify model_dispatch.rs splits landed if step 007 shipped.

5. cargo test --workspace --release passes.

6. Refresh CHANGES.md via ./scripts/gen-changes.sh.

7. Update docs/language-status.md saga timeline + Shipped log with the saga 33 close-out entry. Note the new sibling modules (env_vars, env_models, env_dirs, env_device, env_peer, env_values, env_tags, env_signals; demos_basics, demos_models, demos_training, demos_viz, demos_advanced, demos_mlx; model_apply, model_apply_attention).

8. Quality gates: cargo fmt + cargo clippy + markdown-checker on touched docs.

9. agentrail complete --done.

If the per-step retirements held, this saga should deliver -15+ fails. If they did not, document the gap honestly in the saga close-out commit body, just like saga 32's step 008 did.