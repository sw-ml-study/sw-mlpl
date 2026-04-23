Phase 3 step 004: `demos/embedding_viz.mlpl`
end-to-end.

First runnable embedding-visualization demo. CPU-only.

1. Author `demos/embedding_viz.mlpl`:
   a. Load the `tiny_shakespeare_snippet` preloaded
      corpus + train a byte-level BPE.
   b. Build a small model with an `embed(V, d, seed)`
      layer; e.g. Saga 13 Tiny LM shape at V=280, d=32.
      Pre-train briefly (~100 adam steps) so the
      embedding has learned some structure.
   c. Extract the embedding table. Two paths:
      - Direct: `apply(base.embed, range_ids)` where
        `range_ids = iota(V)` -- this returns the full
        `[V, d]` table.
      - (Alternative, if the embedding name is known)
        read it directly from env via `tok` introspection.
      Use the `apply(embed, iota(V))` pattern in the
      demo -- it is the language-level surface.
   d. `emb_2d = tsne(emb, 30.0, 300, 7)` reduces to
      `[V, 2]`.
   e. `svg(emb_2d, "scatter")` renders the 2-D
      projection.
   f. For the 3-D view: reduce `emb` from d=32 to 3-D
      using the PCA-via-power-iteration pattern from
      the Saga 8 tutorial lesson (document this in a
      comment so the reader knows there is no `pca`
      builtin; composition-only). `svg(emb_3d,
      "scatter3d")`.
   g. Pick a query token id (e.g. the BPE id for " the ")
      and compute `knn(emb, 5)[query_id]` -- the 5
      tokens whose embeddings are closest to the query's.
      Print the result (indices only; the
      tokenizer-reverse-lookup story is nice-to-have
      but not required).

2. Integration test
   `crates/mlpl-eval/tests/embedding_viz_tests.rs`:
   - Cut-down fixture: V=16, d=8, 3 train steps.
   - Assert `tsne` output shape `[16, 2]` with all
     finite values.
   - Assert `knn(emb, 3)` returns `[16, 3]` with every
     index in `[0, 16)` and row i excluding i.
   - Assert the 3-D PCA projection produces a `[16, 3]`
     shape.
   - Rendering calls do not panic (svg returns a
     non-empty String).

3. Run `./target/release/mlpl-repl -f
   demos/embedding_viz.mlpl` manually once; paste the
   final `:wsid` or the knn output into the commit
   message for reproducibility.

4. Add a one-line Demo 9 entry to
   `docs/demos-scripts.md` pointing at the new demo.
   Call out that this is a CPU demo; t-SNE is
   CPU-only today (MLX dispatch for t-SNE's irregular
   inner loop is deferred).

5. Quality gates + `/mw-cp`. Commit demo + test + doc
   entry together. Commit message references Saga 16
   step 004.
