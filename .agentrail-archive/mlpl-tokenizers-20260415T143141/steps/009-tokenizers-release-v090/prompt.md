Phase 4 step 009: Release v0.9.0-tokenizers.

1. New tutorial lessons in apps/mlpl-web/src/tutorial.rs:
   - "Loading Data" (after "Named Axes"): load_preloaded, shuffle, batch, split.
   - "Tokenizing Text" (after "Loading Data"): tokenize_bytes, decode_bytes, train_bpe, apply_tokenizer, decode. Smoke-test every example via mlpl-repl -f.
   - "Experiments" (immediately after "Model Composition" so it can wrap a real training loop): experiment "name" { ... } block, _metric-suffixed capture, :experiments listing.

2. Update docs/are-we-driven-yet.md: move load.csv (row in Datasets section), custom tokenizers (Datasets), and experiment registry (Observability) from CONS/PLAN to HAVE. Any downstream rows that referenced Saga 12 as PLAN get updated.

3. Update docs/saga.md: flip the Future/PLANNED Saga 12 entry to COMPLETE with feature summary.

4. Update docs/status.md: Saga 12 row moves from Planned to Completed with the v0.9.0 target. Downstream saga version targets bump: Saga 13 now v0.10, 14 v0.11, etc.

5. Update docs/plan.md: completed-sagas list gains Saga 12 entry; Future-sequence entry flipped from NEXT to COMPLETE; Start-next section updates to point at Saga 13 (Tiny LM end-to-end).

6. Bump workspace version 0.8.0 -> 0.9.0 in Cargo.toml and the mlpl-web banner strings (help.rs + components.rs x2).

7. Rebuild pages/ via scripts/build-pages.sh with the v0.9.0 banner.

8. Commit via /mw-cp discipline.

9. Tag v0.9.0-tokenizers and push.

10. agentrail complete --done.
