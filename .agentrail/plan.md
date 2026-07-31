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

Steps (draft):

1. engram-in-chain-forward -- eager path: apply_model supports
   ModelSpec::Engram inside a Chain by threading the chain's
   original input ids to Engram layers (embed-first chains: the
   chain input IS the token ids). apply(chain(..., engram_e, ...),
   ids) == manually interleaved apply_engram calls; near-identity
   at init (base output unchanged when engram is fresh). Engram
   outside a chain / chain without leading ids stays a clear error.
2. engram-in-chain-grad -- tape path: apply_model_tape lowers the
   in-chain Engram via engram_tape with the same ids threading, so
   train/adam works on the full model expression. Frozen-base
   test: freeze(base), train N steps -- base params bit-identical,
   engram memory rows move only where addressed; gradcheck vs
   explicit apply_engram composition.
3. engram-stats -- engram_stats(e, ids) builtin: rows_addressed /
   unique_rows / collisions (distinct n-grams sharing a slot),
   mean|max gate activation for a given h (or per-call capture),
   memory row-norm summary (nonzero rows, max norm). Rendered
   :describe-style; unit tests pin the numbers on a fixed seed.
4. tiny-lm-engram-demo -- "Tiny LM + Engram" demo: build the Tiny
   LM base, train briefly, freeze base, insert/attach engram,
   train engram-only, show baseline-vs-engram loss + engram_stats
   panel (gate activity, addressed rows, collisions); demos.toml +
   capabilities/progress-note entries + pinned-count bumps, docs
   refresh, pages rebuild + deploy.
