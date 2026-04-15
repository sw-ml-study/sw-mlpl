Phase 1 step 002: Positional encoding.

Language models need positional information added to token
embeddings (attention is permutation-invariant). Start with
deterministic sinusoidal encoding; add a learned variant
only if the demo benefits.

1. New builtin `sinusoidal_encoding(seq_len, d_model)`
   returning a `[seq_len, d_model]` (labeled `[time, dim]`)
   float array using the standard
   `sin/cos(pos / 10000^(2i/d))` formula. Pure function, no
   params, deterministic.
2. Helper or pattern documented in tests for adding it to a
   `[B, T, d_model]` embedded input via broadcasting on the
   batch axis.
3. Optional (only if needed by the saga 13 demo): builtin
   `positional(max_len, d_model, seed)` that returns a
   `Value::Model` wrapping a `[max_len, d_model]` learned
   table; `apply(pos, [B,T,D])` slices and adds the first
   `T` rows. Skip if sinusoidal works for the demo.
4. TDD:
   - Sinusoidal: shape `[8, 4]` matches a hand-computed
     reference at positions 0 and 1.
   - Determinism: same args produce identical output across
     runs.
   - Broadcasting test: `embed_out + sinusoidal_encoding(T, D)`
     succeeds and labels propagate.
5. Quality gates + `/mw-cp`. Commit message references step 002.
