Phase 3 step 006: Tiny LM training demo (end-to-end).

First end-to-end LM training run in MLPL. Wires every Saga 13
primitive together with the existing tokenizer, dataset, and
experiment surface.

1. Add a small preloaded corpus (compiled-in for the web
   REPL, ~50-100KB plain ASCII). Recommended: a public-domain
   Shakespeare snippet or a deliberately tiny synthetic one.
   Wire it into `load_preloaded("tiny_shakespeare_snippet")`
   (or similar slug).
2. New helper builtin `shift_pairs(ids, block_size)` that,
   given a 1-D `[N]` integer token array, returns
   `(X, Y)` where `X` is `[B, block_size]` token ids and `Y`
   is `[B, block_size]` next-token labels (slice ids by
   `block_size + 1` and split). Document the exact rounding /
   trimming rule. Add unit tests.
3. New `demos/tiny_lm.mlpl`:
       corpus = load_preloaded("tiny_shakespeare_snippet")
       tok    = train_bpe(corpus, 256, 0)
       ids    = apply_tokenizer(tok, corpus)
       (X, Y) = shift_pairs(ids, 32)
       d = 32 ; h = 1 ; V = 256
       model = chain(
         embed(V, d, 0),
         add_positional(32),    # or: + sinusoidal_encoding(32, d)
         residual(chain(rms_norm(d), causal_attention(d, h, 1))),
         residual(chain(rms_norm(d),
                        linear(d, 4*d, 2),
                        relu_layer,
                        linear(4*d, d, 3))),
         rms_norm(d),
         linear(d, V, 4),
       )
       experiment "tiny_lm" {
         train 200 {
           logits = apply(model, X)
           loss   = cross_entropy(logits, Y)
           adam(loss, model, 1e-3, 0.9, 0.999, 1e-8)
           loss_metric = loss
         }
       }
       loss_curve(last_losses)
4. Quantitative outcome: final loss <= 60% of initial loss
   (or stricter if achievable on this corpus). Bake the bound
   into a test that runs a tiny version of the demo (fewer
   steps, smaller vocab) end-to-end.
5. Wire the demo into the web REPL demo list (`demos.rs`).
6. Quality gates + `/mw-cp`. Commit message references step 006.
