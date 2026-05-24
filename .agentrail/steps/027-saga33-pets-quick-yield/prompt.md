Fix the Pets cat-vs-dog (quick) demo's perceived hang in the web UI. Two-axis problem, three-axis fix:

ROOT CAUSES (diagnosed):
1. `scripts/serve.sh` runs `trunk serve` in DEV profile, which produces a ~30x slower WASM bundle than the deployed `pages/` release build. Native release runs the demo in ~14.5s; release WASM is roughly 5x that (~70s, matching the "30-90 seconds" historical experience that ships in `apps/mlpl-web/src/handlers.rs::running_message`); dev WASM is roughly 30x that (~7 minutes -- the user's "spinning after many minutes" report). The github.io live demo (built via `scripts/build-pages.sh` which DOES use --release) is fine; only `serve.sh` is slow.
2. The demo's `train 30 { ... }` is a single MLPL line. The cross-line Timeout(0) yield in `apps/mlpl-web/src/handlers.rs::process_next_eval` cannot help inside one line. Even in release mode the demo blocks the event loop for ~70s, which is long enough for Chrome to put up "Wait / Kill page".

FIXES (do all three):
1. `scripts/serve.sh` defaults to release. Add a `--dev` toggle (or document the fast-rebuild env var) so the inner dev loop can opt back into the slower-binary / faster-recompile profile when actively debugging the front-end. The default should be the one users want by default, which is release.
2. Re-chunk the `PETS_CAT_VS_DOG_QUICK` demo source in `apps/mlpl-web/src/demos_vit.rs` into multiple `train K {...}` blocks bound to a repeating outer step, exactly like `PETS_PREDICT_GALLERY` already does ("30 full-batch Adam steps on 16 images, run as 6 separate train blocks of 5 steps each"). Adam state persists across train blocks via env.optim_state, so this is mathematically equivalent. Cross-line yields then kick in between chunks; the spinner caption updates between chunks; and the user sees an incremental loss curve. Suggested split: `train 5` x 6 = 30 steps, or `train 6` x 5 -- whichever balances responsiveness vs. chunk count.
3. Update the spinner caption logic in `apps/mlpl-web/src/handlers.rs::running_message` if the chunked source no longer triggers the "train" branch nicely. Specifically: the train-block branch should still fire for the inner `train 5 {...}` lines, OR a new branch for the outer `repeat 6` should produce a per-chunk progress hint. Also consider showing the chunk index ("chunk 3/6") in the spinner -- it's the simplest progress signal we can ship without async eval.

NON-GOAL: do NOT introduce Web Workers in this step (docs/worker-threads.md tracks that as a separate, much larger project). The chunked-source + release-default approach is the surgical fix that uses infrastructure already in the repo.

QUALITY GATES:
1. After the demo source change, rebuild pages/ via `scripts/build-pages.sh` so the github.io live demo picks up the new chunked source. Commit pages/ in the same commit.
2. cargo test --release for any mlpl-eval / mlpl-web tests that touch the demos table.
3. cargo clippy --workspace --all-targets --all-features -- -D warnings.
4. cargo fmt --all -- --check.
5. markdown-checker for any md changes.
6. sw-checklist net-negative on both axes.

VERIFICATION:
- Run `scripts/serve.sh` (now defaults to release). Open http://localhost:9957/. Click "Pets: cat vs dog (quick)". Confirm:
  (a) The browser stays responsive throughout (no Chrome "Wait / Kill" dialog).
  (b) Loss values appear incrementally (one chunk at a time), not all at the end.
  (c) Total wall time is comparable to the historical 30-90 seconds.

DELIVERABLE: a single saga step that retires the perceived-hang regression. Future Web Worker work tracked separately as the "real" solution (the comment in handlers.rs around line 213 still applies).
