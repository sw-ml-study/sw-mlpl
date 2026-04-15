Phase 2 step 005: Sampling -- multinomial + top-k.

Generation needs sampling from a distribution, not just
argmax.

1. New builtin `sample(logits, temperature, seed)`:
   - `logits`: 1-D `[V]` float array (one position's logits).
   - `temperature`: f64. `0.0` collapses to `argmax(logits)`.
     Otherwise compute `softmax(logits / temperature)` and
     draw one categorical sample.
   - `seed`: u64. Same seed + same logits + same temperature
     => same draw (use a deterministic PRNG; reuse whatever
     `random` / `randn` use elsewhere).
   - Returns a scalar integer token id.
2. New builtin `top_k(logits, k)`:
   - Returns a 1-D `[V]` float array with all but the top-k
     entries set to `-inf` (or a very large negative). Pure
     function, no randomness.
3. Composition pattern (no new combined builtin):
   `sample(top_k(logits, k), temperature, seed)`.
4. TDD:
   - Determinism: same `(logits, temp, seed)` returns the
     same id across calls.
   - `temperature=0` matches `argmax(logits)` on several
     random `[V]` inputs.
   - `top_k(logits, 1)` then `sample` returns the argmax
     regardless of temperature.
   - Distribution check: 10000 samples at high temperature
     approximate `softmax(logits)` within a chi-square or
     KL-div tolerance.
5. Quality gates + `/mw-cp`. Commit message references step 005.
