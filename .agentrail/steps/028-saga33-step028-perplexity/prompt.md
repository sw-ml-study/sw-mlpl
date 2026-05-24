Add a `perplexity(logits, targets)` builtin convenience function. Update the glossary entry and add perplexity computation to LM-style demos.

DELIVERABLES:

1. NEW BUILTIN: `perplexity(logits, targets)` in `crates/mlpl-runtime/src/ml_builtins.rs`. Implementation: `exp(cross_entropy(logits, targets))`. Reuses the existing `builtin_cross_entropy` -- compute the scalar CE loss then call `exp` (or directly call `builtin_cross_entropy` and exp the result). Add to NAMES: `["softmax", "one_hot", "sinusoidal_encoding", "cross_entropy", "perplexity"]`. Hook into the `try_call` match. Same arg shape as cross_entropy: `[N, V]` (or `[B, T, V]`) logits + length-N (or `[B, T]`) integer targets.

2. UPDATE GLOSSARY (docs/glossary.md): the "Perplexity" entry should reflect that MLPL now ships the convenience function. Replace the current entry with:

   ## Perplexity

   The exponentiated cross-entropy of a language model on a held-out corpus:
   `exp(cross_entropy_loss)`. Standard LM evaluation metric. Lower is better.
   MLPL ships `perplexity(logits, targets)` as a convenience -- it returns
   `exp(cross_entropy(logits, targets))` in one call.

3. ADD A UNIT TEST in `crates/mlpl-runtime/tests/` (or extend an existing tests file) confirming `perplexity` equals `exp(cross_entropy(...))` on a small fixture.

4. UPDATE DEMOS: search for demos that compute a cross-entropy loss on an LM-like task and add a perplexity readout. Likely candidates:
   - `apps/mlpl-web/src/demos_lm.rs::TINY_LM` -- the Tiny LM demo trains a small LM with cross_entropy. Add `perplexity(...)` after training to show the held-out (or just the training-loss) perplexity as a scalar with a 1-line comment.
   - `apps/mlpl-web/src/demos_lm.rs::TINY_LM_GENERATE` -- if it already computes CE, add perplexity. If not, skip.
   - On-disk demos in `demos/*.mlpl` that train LMs (e.g., `demos/tiny_lm.mlpl` if it exists). Add perplexity readout there too.

   Each demo update is one new line: `perplexity_score = perplexity(logits, targets) # exp(CE)`. The intro/takeaway text should be tweaked to mention perplexity as the canonical LM metric where natural; do not bloat the text.

5. UPDATE LANGUAGE-REFERENCE if `docs/lang-reference.md` (or equivalent) lists builtins with one-line docs. Add a `perplexity(logits, targets)` row matching the cross_entropy row's style.

6. Rebuild pages/ since apps/mlpl-web changed. Commit pages/ in the same commit.

OUT OF SCOPE:
- A separate "held-out evaluation" pipeline. The demo additions show perplexity on the SAME data the LM trained on (a training-perplexity readout). True held-out perplexity needs a train/test split, which is a bigger demo redesign.
- Bits-per-byte or per-token-perplexity variants. Standard `exp(CE)` only.
- A grad path through `perplexity` (it's a metric, not a loss).

QUALITY GATES:
1. `cargo test --release` -- new unit test + all_demos_smoke still passes with the updated demos.
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
3. `cargo fmt --all -- --check`.
4. `markdown-checker docs/glossary.md` (and any other docs touched).
5. `sw-checklist` net-negative on FAILs and warnings.
6. Push after commit.
