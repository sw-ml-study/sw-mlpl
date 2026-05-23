Tech-debt saga step 008 (FINAL, use --done): final ratchet + update docs.

1. Final sw-checklist pass. Quote the start-of-saga (151 fails / 452 warnings) and end-of-saga counts in the commit body. Verify halved-on-both-axes target was met (or document why if not -- with concrete reason, not boilerplate exception).

2. Run the full workspace test sweep: cargo test --workspace --release.

3. Refresh CHANGES.md via ./scripts/gen-changes.sh.

4. Update docs/language-status.md: add a 'Saga 32 (tech-debt-paydown) closed' entry at the top of the Shipped log. Clear the active-saga row.

5. Update docs/code_metrics.md if any new file-naming conventions or refactoring patterns emerged that future agents should know about.

6. Quality gates: cargo fmt + cargo clippy + markdown-checker on touched docs.

7. agentrail complete --done.