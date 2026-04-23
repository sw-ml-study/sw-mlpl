Phase 3 step 006: web tutorial lesson + `docs/using-lora.md`.

1. Add a "LoRA Fine-Tuning" tutorial lesson to the web
   REPL, in `apps/mlpl-web/src/lessons.rs` (the module we
   extracted in Saga 20 step 006). Mirror the Saga 20
   "Neural Thickets" lesson shape:
   - Short narrative (~4-5 sentences) explaining LoRA:
     frozen base + two small adapter matrices A, B;
     zero-init on B; the `alpha/rank` scaling;
     compositionality with `freeze`.
   - Examples that build up: `m = linear(4, 8, 0);
     freeze(m); lora_m = lora(m, 2, 4.0, 7)`; show the
     new params (`:describe lora_m`); forward-identity
     check (`apply(lora_m, X) == apply(m, X)` before
     training); one Adam step; forward changes.
   - A `svg(B_adapter, "heatmap")` or `svg(A_adapter,
     "heatmap")` at the end showing the learned adapter
     after a few steps so the lesson produces a visible
     artifact.
   - A `try_it` that asks the user to rerun with rank=1
     vs rank=4 and compare the forward delta.
   - Keep the example sizes small (V=8, d=4, rank=2) so
     it renders interactively in WASM.

2. Rebuild `pages/` via `./scripts/build-pages.sh` and
   commit the refreshed bundle hash alongside the source
   changes (per CLAUDE.md live-demo policy / project
   memory `feedback_pages_rebuild.md`).

3. Write `docs/using-lora.md` covering:
   - Why LoRA: context + the "freeze + small adapters"
     pattern, in one paragraph.
   - The surface: `freeze(m)`, `unfreeze(m)` (if it
     shipped in step 001), `lora(m, rank, alpha, seed)`.
     One paragraph each.
   - The zero-init-B rationale and the alpha/rank scaling
     convention.
   - The demo walkthrough (CPU + MLX variants).
   - Measured MLX-vs-CPU numbers from step 005.
   - Parity testing.
   - The deferred follow-up surface: QLoRA / 4-bit
     quantization, selective layer attachment (LoRA only
     on attention projections), `merge_lora(m)` for
     inference deployment, multi-adapter composition /
     routing, real pretrained checkpoints.

4. `markdown-checker -f "docs/using-lora.md"` (ASCII-only).

5. Quality gates + `/mw-cp`. Commit message references
   Saga 15 step 006.
