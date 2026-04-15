Phase 3 step 007: Generation loop + attention-map visualization.

With the trained model from step 006, demonstrate the two
things every LM user expects: generate text from a prompt,
and look inside the attention.

1. Add `demos/tiny_lm_generate.mlpl`:
   - Train (or rebuild quickly) a tiny model as in step 006.
   - Define a generation loop using existing constructs
     (`repeat N { ... }` or a small new helper):
       prompt_ids = apply_tokenizer(tok, "to be ")
       generated  = prompt_ids
       repeat 40 {
         logits     = apply(model, generated)        # [1, T, V]
         last       = logits[:, -1, :]               # [1, V]
         next_id    = sample(top_k(last, 40), 0.8, step)
         generated  = concat(generated, next_id)     # along time
       }
       text_out = decode(tok, generated)
   - If MLPL lacks `concat` along an axis or negative-index
     slicing, add the minimum needed (prefer extending an
     existing builtin over a new one). Document any addition
     in the commit message.
2. Attention-map viz: extract one forward pass's attention
   weights (`[heads, T, T]`) for a single input. Either:
   (a) Add `attention_weights(model, X)` read-only builtin
       that runs the model and returns the weights tensor
       from the (single) attention layer encountered, or
   (b) Add a debug capture flag on the model that stores
       weights in `Environment` for inspection.
   Render via `svg(weights[0], "heatmap")` to produce a
   per-head attention triangle.
3. TDD:
   - Generation loop produces a string of the expected length
     when given a fixed seed. Two runs with the same seed
     match byte-for-byte.
   - Attention weights are lower-triangular when causal mask
     is on (zeros / near-zeros above the diagonal).
4. Wire `tiny_lm_generate.mlpl` into the web REPL demo list.
5. Quality gates + `/mw-cp`. Commit message references step 007.
