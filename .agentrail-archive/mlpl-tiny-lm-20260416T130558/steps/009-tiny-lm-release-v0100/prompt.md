Phase 5 step 009: Release v0.10.0.

Cut the Tiny LM release.

1. Bump workspace version to `0.10.0` (workspace `Cargo.toml`
   and any per-crate versions that track it).
2. Update `docs/status.md` Saga 13 row to `[x] | v0.10.0`.
3. Finalize `docs/saga.md` Saga 13 section with the shipped
   feature list (mirror the Saga 12 retrospective shape).
4. Finalize `docs/milestone-tiny-lm.md` if step 008 left it
   skeletal.
5. Verify the live demo build:
   - `scripts/build-pages.sh` still produces a clean diff (or
     a no-op).
   - Commit any rebuilt `pages/` if 008 missed it.
6. Quality gates + `/mw-cp`:
   - `cargo test`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo fmt --all -- --check`
   - `markdown-checker -f "**/*.md"`
   - `sw-checklist`
7. Detailed release commit:
   `release(v0.10.0): Saga 13 COMPLETE -- Tiny LM end-to-end`
   listing primitives shipped (`embed`, `sinusoidal_encoding`,
   `causal_attention`, `cross_entropy`, `sample`, `top_k`,
   `shift_pairs`, `attention_weights` if added) plus demos
   and tutorial lessons.
8. `git tag v0.10.0` and `git push origin main --tags`.
9. Verify pages workflow: `gh run list --workflow=pages.yml
   --limit 1`.
10. `agentrail complete --done` after the release commit and
    push are confirmed.
