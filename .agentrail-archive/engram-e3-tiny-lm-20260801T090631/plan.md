# Saga E3: engram-tiny-lm (Engram inside Tiny-LM blocks + stats)

Per docs/engram-sagas-plan.md (E3) and decisions D2/D3. E1 gave the
hash/gather primitives (mlpl-engram-core); E2 gave the trainable
engram(...) model, apply_engram forward, and tape lowering. E3 puts
the Engram INSIDE a Tiny-LM block (the "AfterAttention" insertion),
trains it against a frozen base, adds engram_stats, and ships the
Tiny LM + Engram demo with a stats panel. CPU only (D2); MLX comes
with E4/E5.

State of the repo (verified 2026-07-31):

- ModelSpec::Engram exists (mlpl-eval-core/src/model.rs) but BOTH
  apply_model (mlpl-eval-models/src/model_apply.rs) and
  apply_model_tape (mlpl-models-tape/src/apply.rs) return
  Unsupported for it -- Engram is only reachable via the explicit
  apply_engram(e, h, ids) surface. The chain integration is the gap.
- The Tiny LM is MLPL source built from combinators:
  chain(embed(V,d,0), residual(chain(rms_norm(d),
  causal_attention(d,h,1))), rms_norm(d), linear(d,V,4)).
  Inserting Engram after the attention residual means the chain
  apply must thread the ORIGINAL token ids (the chain's input,
  which embed consumes) down to any Engram layer in the chain.
- freeze(m)/unfreeze(m) + adam/momentum_sgd frozen-skip already
  work (grad_optim.rs) -- frozen-base training reuses them as-is.
- No engram_stats builtin exists anywhere; net-new.
- Demo pipeline: demos.toml (category = "Engram" group exists),
  optional [[capabilities]]/[[progress_notes]] tables, pinned
  counts in mlpl-web-demos/tests/metadata_codegen.rs.

Steps:

1. engram-in-chain-forward -- DONE (commit 8a445abf). apply_model
   supports ModelSpec::Engram inside a Chain by threading the
   chain's original input ids; chain() args resolve bound model
   identifiers; near-identity + manual-composition-equality +
   error-surface tests in engram_chain_tests.rs.
2. engram-in-chain-grad -- tape path: apply_model_tape lowers the
   in-chain Engram via engram_tape with the same ids threading, so
   train/adam works on the full model expression. Frozen-base
   test: freeze(base), train N steps -- base params bit-identical,
   engram memory rows move only where addressed; gradcheck vs
   explicit apply_engram composition.
3. md-ascii-cleanup -- tech-debt spike (user-queued 2026-07-31):
   fix hand-owned non-ASCII markdown (AGENTS.md, 8 docs/ files,
   5 books/ files), teach gen-changes.sh to transliterate old
   commit subjects, reconcile the markdown-checker /
   sw-markdown-checker binary-name mismatch. Out of scope:
   .agentrail-archive, vendor/, the agentrail-managed CLAUDE.md
   block (upstream agentrail emits em dashes -- flag it).
4. cuda-target-gating -- build-matrix spike (user-queued
   2026-07-31): move candle-core behind
   [target.'cfg(linux+x86_64)'.dependencies] in the cuda crates so
   --features cuda compiles as a stub on macOS (mirror of mlx on
   Linux) and --all-features clippy works on both hosts. Linux
   behavior unchanged; verify on the CUDA box after the move back
   if not verifiable this side.
5. engram-stats -- engram_stats(e, ids) builtin: rows_addressed /
   unique_rows / collisions (distinct n-grams sharing a slot),
   mean|max gate activation for a given h (or per-call capture),
   memory row-norm summary (nonzero rows, max norm). Rendered
   :describe-style; unit tests pin the numbers on a fixed seed.
6. tiny-lm-engram-demo -- "Tiny LM + Engram" demo: build the Tiny
   LM base, train briefly, freeze base, insert/attach engram,
   train engram-only, show baseline-vs-engram loss + engram_stats
   panel (gate activity, addressed rows, collisions); demos.toml +
   capabilities/progress-note entries + pinned-count bumps, docs
   refresh, pages rebuild + deploy.
