Tech-debt saga step 001: split mlpl-eval crate.

mlpl-eval has 42 modules (max 7). This is the worst Crate-Module-Count FAIL in the workspace and the structural root of many Module-Fn-Count + Function-LOC fails downstream.

Split into a facade + 3 new sibling crates per docs/saga-tech-debt-paydown.md:

1. mlpl-eval (facade) -- keep the eval core: eval.rs, eval_program.rs, eval_for.rs, eval_loop.rs, eval_intercepts.rs, eval_ops.rs, eval_reduce.rs, eval_script.rs, env.rs, error.rs, value.rs, device.rs, interrupt.rs. ~13 modules. lib.rs becomes a facade that pub use's everything from the new sibling crates so downstream imports (mlpl-cli, mlpl-serve, mlpl-repl, mlpl-web, mlpl-bench) are unchanged.

2. mlpl-eval-model (new) -- ModelSpec + every operation on it: model.rs, model_clone.rs, model_dispatch.rs, model_embed_table.rs, model_estimate.rs, model_feasibility.rs, model_freeze.rs, model_lora.rs, model_perturb.rs, model_tape.rs.

3. mlpl-eval-grad (new) -- autograd + gradient accumulation: grad.rs, grad_optim.rs, tag_propagate.rs, tag_render.rs, auto_tag.rs.

4. mlpl-eval-data (new) -- loaders, tokenizers, telemetry: bpe.rs, loader.rs, tokenizer.rs, type_errors.rs, metric_sink.rs, experiment.rs, inspect.rs, inspect_groups.rs, result_ops.rs, llm_dispatch.rs, pets_tiny.rs, fetch_dataset.rs, image_io.rs.

Process:
- For each new crate: create crates/<name>/{Cargo.toml, src/lib.rs}, register in workspace Cargo.toml, move the source files in (git mv), update use paths inside the moved files (s/crate::X/crate::Y::X/ where needed -- usually still 'crate::' if the dep is in the same new crate, else 'mlpl_eval_model::' etc.).
- mlpl-eval's lib.rs becomes a facade: 'pub use mlpl_eval_model::*' etc.
- Test that EVERY existing test still compiles + passes unchanged.
- cargo clippy --workspace --all-targets --all-features -- -D warnings.
- cargo fmt.
- Run sw-checklist BEFORE the split and AFTER. Commit body must quote both. Expected: -1 Crate-FAIL on mlpl-eval, +3 Crate-Module-Count entries (one per new crate, but each new crate should have ~10 modules so they each rack up ~1 Module-Fn-Count + maybe 1 Crate-Module-Count). Net target: -1 to -3 FAILs.

Quality gates: workspace tests green, clippy green, fmt green, sw-checklist net-negative on BOTH fails AND warnings vs HEAD~1.