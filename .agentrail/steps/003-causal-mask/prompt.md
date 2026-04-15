Phase 1 step 003: Causal masking for attention.

Existing `attention(d_model, heads, seed)` lets every position
attend to every other position. Language models need a causal
mask so position `t` cannot peek at positions `>t`.

1. Pick one of:
   (a) New builtin `causal_attention(d_model, heads, seed)`
       constructing a model identical to `attention` but with
       a causal-mask flag set. Recommended for clarity.
   (b) Optional bool/string parameter on `attention`. Avoid if
       it complicates the tape-lowered path.
2. In the tape-lowered (heads=1) `attention` forward, after
   computing `Q K^T / sqrt(d)`, before softmax, add a
   broadcastable `[T, T]` mask whose upper triangle is a large
   negative (e.g. -1e9). Reuse the existing tape primitives so
   gradients still flow.
3. TDD:
   - Forward: `[B=1, T=3, d=4]` causal attention. Perturbing
     position 1 input must not change position-0 output (within
     epsilon). Without the mask, it would.
   - Gradcheck against finite differences on a tiny case so
     the masked path stays differentiable.
   - `params(causal_attention(d, 1, seed))` returns the same
     param set as `params(attention(d, 1, seed))`.
4. Quality gates + `/mw-cp`. Commit message references step 003.
