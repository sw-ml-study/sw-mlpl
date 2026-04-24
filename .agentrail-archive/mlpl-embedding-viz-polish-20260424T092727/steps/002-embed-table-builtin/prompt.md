Phase 1 step 002: `embed_table(model) -> [vocab, d_model]`
builtin.

Extract an embed layer's weights by walking a
`ModelSpec` tree. Closes the Saga 16 gap where
training a full `chain(embed, transformer_block,
head)` had no way to pull the learned embedding table
back out at the source level.

1. **Signature**.
   - `model` -- bare model identifier (looked up in
     `env.models`) or any expression that evaluates
     to `Value::Model`. Same arg shape as
     `clone_model` / `freeze` / `lora`.
   - Returns rank-2 `[vocab, d_model]` (the embed
     layer's table).
   - If the model contains no Embedding layer:
     `EvalError::Unsupported("embed_table: model
     contains no Embedding layer")`.

2. **Tree walk semantics**.
   - Match on `ModelSpec`:
     - `Embedding { table, .. }`: return
       `env.get(table).unwrap().clone()`.
     - `Chain(children)`: iterate children in order;
       return the first child whose recursive walk
       returns `Some(table)`.
     - `Residual(inner)`: recurse into inner.
     - `Linear`, `Activation`, `RmsNorm`,
       `Attention`, `LinearLora`: no embedding;
       return `None`.
   - **First-match wins.** If a model somehow has
     two Embedding layers (unusual -- multi-
     embedding stacks are not a shipped pattern),
     `embed_table` returns the first one found in
     depth-first left-to-right order. Document in
     the contract.

3. **Module**: new
   `crates/mlpl-eval/src/model_embed_table.rs`.
   Small module: `eval_embed_table(args, env)` pub
   + `find_embedding_table(spec, env)` recursive
   helper. 2 fns, well under the 7-fn budget.
   Design for budgets up front.

4. Wire into `eval.rs` as a new FnCall branch next
   to `freeze` / `unfreeze`. Returns `Value::Array`
   so the call sits in expression position:
   `table = embed_table(model)`.

5. **Contract** at
   `contracts/eval-contract/embed-table.md`:
   - Signature, first-match semantics, error cases.
   - Non-goals: no multi-embedding support (first-
     match only); no path-selector variant like
     `embed_table(model, "encoder.embed")`; no sub-
     layer introspection beyond Embedding.

6. **TDD** (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-eval/tests/embed_table_tests.rs`:
   - Standalone embed:
     `e = embed(V, d, 0); t = embed_table(e)`
     returns a `[V, d]` matrix; values match
     `apply(e, iota(V))` elementwise.
   - Embed inside a chain:
     `m = chain(embed(V, d, 0), linear(d, V, 1));
     t = embed_table(m)` -- returns the embed's
     table (the W inside the chain's Linear is
     NOT returned).
   - Embed inside a Residual:
     `m = residual(embed(V, d, 0))` -- same.
   - Nested chain:
     `m = chain(chain(embed(V, d, 0), ...), ...)`
     -- walks down to find it.
   - After training, `embed_table` returns the
     UPDATED weights: train a chain that includes
     an embed; compare `embed_table` before and
     after `train N { adam(...) }`; values differ.
   - Model with no embed:
     `embed_table(linear(3, 4, 0))` errors with
     the expected message.
   - Wrong arity.
   - Non-model argument (e.g., an array var).

7. Quality gates + `/mw-cp`. Commit message
   references Saga 16.5 step 002.
