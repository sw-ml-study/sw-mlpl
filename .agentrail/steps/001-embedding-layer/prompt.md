Phase 1 step 001: Token embedding layer.

Add a token embedding to the Model DSL so language models can
map integer token ids to learned dense vectors.

1. New builtin `embed(vocab_size, d_model, seed)` returning a
   `Value::Model` (mirror how `linear` constructs and registers
   itself). The model owns one parameter: a `[vocab, d_model]`
   labeled-shape table named so `params(model)` walks it, and
   `adam(loss, model, ...)` updates it.
2. Extend `apply(emb, tokens)` so when the model is an
   embedding and `tokens` is an integer-typed array of shape
   `[B, T]` (or `[B*T]`), the result is `[B, T, d_model]` (or
   `[B*T, d_model]`) float, with rows gathered from the table.
3. Tape-lower the gather so `grad(...)` flows back into the
   embedding table (an index-add into the gradient buffer).
4. TDD:
   - Unit: `(vocab=5, d=3)` table, `tokens=[0,2,4]` -- output
     row 0 equals `table[0]`, etc.
   - Gradcheck against finite differences on a small loss like
     `sum(apply(emb, tokens) * ones)`.
   - `params(emb)` returns one `[5, 3]` parameter; one Adam
     step on the toy loss reduces it.
5. Wire `:describe emb` to print `embed[vocab=5, d=3]`.
6. Quality gates + `/mw-cp`. Commit message references step 001.
