# Embedding-viz Polish Milestone (Saga 16.5, v0.14.1)

## Why this exists

Saga 16 shipped the embedding-visualization surface
(`pairwise_sqdist`, `knn`, `tsne`, `svg(...,
"scatter3d")`) plus `demos/embedding_viz.mlpl`. It
deliberately punted on two loose ends that the shipped
doc explicitly flagged:

1. **PCA is composition-only, not a builtin.** The
   Saga 8 tutorial lesson shows power-iteration +
   deflation for top-1 PCA; extending to top-k
   requires iterated deflation that's awkward to
   write at the source level every time.
2. **No way to extract an embed layer's weights from
   a trained chain.** `chain(emb, transformer_block,
   head)` trains an embedding internally, but the
   source level has no `embed_table(model)` to pull
   the `[V, d]` matrix back out. The Saga 16 demo
   worked around this by training a standalone
   `embed` model + MSE-to-target, but the natural
   flow (train full model, inspect embed sublayer)
   needs a builtin.

Saga 16.5 ships both: a `pca(X, k)` builtin that
wraps power iteration + deflation into a one-liner,
and an `embed_table(model)` builtin that walks a
ModelSpec tree and returns the first Embedding node's
[vocab, d_model] table.

Neither is load-bearing for Saga 19 (LLM-as-tool
REST) or Saga 21 (CLI server), so this is a short
polish pass before the next real saga.

## Non-goals (deferred, AGAIN)

- **UMAP.** Separate nonlinear reducer; full
  implementation is a t-SNE-scale algorithm.
  Overlaps with t-SNE for marginal user value.
  Defer until a concrete use case surfaces.
- **Interactive 3-D scatter.** Needs a JS rotator;
  separate frontend saga.
- **MLX dispatch for t-SNE.** Inner loop doesn't
  vectorize cleanly; separate optimization pass.
- **RAG pipeline.** Needs LLM inference path (Saga
  19).
- **`merge_lora` / QLoRA / etc.** Saga 15 deferrals,
  not Saga 16.

## Quality requirements (every step)

Identical to Saga 16; `docs/sw-checklist-patterns.md`
is the decomposition reference. Design for budgets
up front.

## Phase 1 -- two builtins (2 steps)

### Step 001 -- `pca(X, k) -> Y [N, k]`
Top-k PCA via power iteration + deflation, wrapped
into a single builtin.

1. **Signature**.
   - `X` rank-2 `[N, D]`.
   - `k` positive integer, `1 <= k <= D`.
   - Returns rank-2 `[N, k]` of centered-and-
     projected data. The k principal components
     themselves (eigenvectors) are NOT returned as a
     separate output; users who need them can run
     the composition pattern directly. Keeping a
     single rank-2 return keeps the surface simple.

2. **Algorithm**.
   a. Center: `Xc = X - col_means(X)`.
   b. `Cov = Xc^T Xc / N`.
   c. For `i in 0..k`:
      - Power-iterate `v` starting from `[1, 0,
        ...]` (or a seeded randn) for M iterations
        (50 is fine; document the choice).
      - `lambda = v^T Cov v`.
      - Record `v` as component i.
      - Deflate: `Cov = Cov - lambda * (v @ v^T)`.
   d. Stack components into `V [k, D]`; return
      `Xc @ V^T [N, k]`.

3. **Module**: new
   `crates/mlpl-runtime/src/pca_builtin.rs` (small;
   the algorithm is short enough to fit in one
   module without further splits).

4. **Contract**:
   `contracts/eval-contract/pca.md` -- signature,
   algorithm, the "Y is the projected data, not the
   components themselves" choice, error cases, non-
   goals (no UMAP, no sparse PCA, no SVD-based PCA).

5. **TDD** in
   `crates/mlpl-eval/tests/pca_tests.rs`:
   - 2-D to 1-D dimensionality reduction on an
     obviously-anisotropic fixture (e.g.,
     `X = matmul([...], [[1, 0.3], [0, 0.1]])`
     stretches along axis 0); assert the 1-D
     projection captures the majority of variance.
   - Shape preservation: any `[N, D]` input +
     `1 <= k <= D` gives `[N, k]` output.
   - `k == D` roughly preserves variance (sum of
     variances in the projection equals total input
     variance, within numerical tolerance).
   - Determinism: two calls with identical input
     produce bit-identical output.
   - Error cases: rank != 2, k == 0, k > D, non-
     finite X.

### Step 002 -- `embed_table(model) -> [vocab, d_model]`
Extract an embed layer's weights by walking a
`ModelSpec` tree.

1. **Signature**.
   - `model` -- model identifier (bare Ident
     bound in `env.models`) or expression evaluating
     to `Value::Model`. Same argument shape as
     `clone_model` / `freeze` / `lora`.
   - Returns rank-2 `[vocab, d_model]` matrix.

2. **Tree walk**.
   - If the model is `ModelSpec::Embedding { table,
     vocab, d_model, .. }`: return
     `env.get(table).unwrap()`.
   - If `Chain(children)`: recurse into children in
     order; return the first match.
   - If `Residual(inner)`: recurse into inner.
   - If any other variant (Linear, Activation,
     RmsNorm, Attention, LinearLora): skip.
   - If no Embedding found anywhere: return an
     `EvalError::Unsupported("embed_table: model
     contains no Embedding layer")`.

3. **Module**: new
   `crates/mlpl-eval/src/model_embed_table.rs` (one
   pub fn + one recursive helper; under the 7-fn
   budget).

4. **Contract**:
   `contracts/eval-contract/embed-table.md` --
   signature, first-match semantics, error cases,
   non-goals (no multi-embedding support; if a user
   somehow has two embeddings in a chain, only the
   first is returned).

5. **TDD** in
   `crates/mlpl-eval/tests/embed_table_tests.rs`:
   - Standalone embed: `embed_table(embed(V, d, 0))`
     returns `[V, d]`; values match
     `apply(emb, iota(V))`.
   - Chain with embed at position 0:
     `embed_table(chain(embed(V, d, 0), ...))`
     returns the same matrix as a standalone embed
     with the same seed would (bit-identical since
     embed's init is seed-deterministic).
   - Embed inside Residual: same.
   - Model with no embed (e.g., a bare Linear):
     error with the expected message.
   - Nested chains:
     `embed_table(chain(chain(embed(...), ...),
     ...))` walks down to find it.

## Phase 2 -- docs + release (1 step)

### Step 003 -- update demos/docs/tutorial + release v0.14.1
1. **Update `demos/embedding_viz.mlpl`** to use both
   builtins:
   - Replace the column-selector matmul 3-D
     projection with `emb_3d = pca(table, 3)`. One
     line instead of the reshape + matmul pair.
   - Keep the standalone-embed + MSE-to-target
     training story (no change; `embed_table` is
     demonstrated in the tutorial lesson below).

2. **Update `docs/using-embeddings.md`**:
   - "PCA is a composition pattern" section -> "PCA
     is shipped as a builtin; the composition
     pattern still works and is a good pedagogical
     reference." Link to the Saga 8 lesson as the
     under-the-hood explanation.
   - New "Extracting embed-layer weights" section
     describing `embed_table(model)` + its use from
     within a full trained chain.
   - "Not shipped" list: remove `pca(X, k)` +
     `embed_table(model)`; keep the other items.

3. **Update the "Embedding exploration" web REPL
   lesson** (`apps/mlpl-web/src/lessons_advanced.rs`):
   Add 1-2 example lines showing `pca(X, 2)` and
   `embed_table(emb)` on a tiny model. Keep the
   lesson concise (already at the file's lesson
   count; no new lessons, just new examples).

4. **Rebuild `pages/`**.

5. **Release v0.14.1**:
   - Bump `Cargo.toml` `workspace.package.version`
     `0.14.0 -> 0.14.1`. Patch-level bump because
     the surface adds two small convenience
     builtins; no breaking changes.
   - Add a v0.14.1 section to `CHANGELOG.md` above
     v0.14.0: the two builtins + demo/docs/lesson
     updates.
   - Short Saga 16.5 entry in `docs/saga.md` above
     Saga 16 (or folded as an "Addendum" under
     Saga 16's retrospective; pick during
     implementation).
   - Mark Saga 16.5 complete in `docs/status.md`;
     remove the `pca + embed_table` deferred-
     follow-up row.
   - Tag `v0.14.1` locally; confirm before pushing
     per the v0.12.0 / v0.13.0 / v0.14.0 cadence.
   - `/mw-cp` quality gates.

## Dependency graph

```
001 pca builtin    002 embed_table builtin
    \__________ orchestration __________/
                        |
             003 demos/docs/release v0.14.1
```

Steps 001 and 002 are independent and could run in
parallel; sequential here for commit-history
simplicity.
