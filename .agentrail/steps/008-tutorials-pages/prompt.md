Phase 4 step 008: Tutorial lessons + pages rebuild.

Make Saga 13 visible in the web REPL tutorials and ship the
docs.

1. Add two web REPL tutorial lessons (in the same place
   prior tutorials live -- mirror the structure used by
   "Loading Data" / "Tokenizing Text" / "Experiments" from
   Saga 12):
   - **"Language Model Basics"**: walks from `embed` through
     `sinusoidal_encoding` (or `add_positional`),
     `causal_attention`, a forward pass, and `cross_entropy`.
     Should run in <2s in the browser.
   - **"Training and Generating"**: a stripped-down version
     of `tiny_lm.mlpl` (smaller corpus / fewer steps) wrapped
     in `experiment "tutorial_tiny_lm"`, followed by a 20-token
     generation loop and the attention heatmap.
2. Add `docs/milestone-tiny-lm.md` retrospective summarizing
   what shipped (mirror `milestone-tokenizers.md` shape).
3. Update `docs/status.md` so the Saga 13 row reads
   `[~] in progress` (step 009 will flip it to `[x]`).
4. Update `docs/saga.md` with a Saga 13 in-progress section
   (final wording lands in step 009).
5. Rebuild `pages/` via `scripts/build-pages.sh`. Commit
   source + built artifact in the SAME commit so the live
   demo shows the new lessons after push.
6. ASCII-only markdown (`markdown-checker -f "**/*.md"`).
7. Quality gates + `/mw-cp`. Commit message references step 008.
