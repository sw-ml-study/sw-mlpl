# Tiny Language Model End-to-End Milestone (Saga 13, v0.10.0)

## Why this exists

Saga 13 is the first saga that proves the MLPL platform thesis
end-to-end: a user writes a handful of lines of MLPL, and a
small transformer-based language model trains, generates text,
and visualizes its own attention -- all on CPU, with reproducible
experiment tracking. Every prior saga was preparation:

- Saga 9 gave us reverse-mode autograd.
- Saga 10 gave us optimizers and the `train { }` loop.
- Saga 11 gave us `chain` / `residual` / `attention` / `rms_norm`
  for composing models.
- Saga 11.5 added shape checking via labeled axes.
- Saga 12 added file IO, batched dataset iteration, a byte-level
  BPE tokenizer, and `experiment "name" { }` for tracking runs.

Saga 13 fills the six primitive gaps that still stood between the
platform and a working language model -- token embeddings,
positional information, causal masking, cross-entropy loss,
multinomial / top-k sampling, and a generation loop -- and then
demos the whole stack with a tiny transformer LM trained on a
small corpus.

## Non-goals

- **No new backends.** CPU only. MLX / CUDA are Sagas 14 and 17.
- **No pretraining claims.** Target was ~1-5M params on a
  ~100KB corpus -- enough to see loss drop and coherent-ish
  snippets, not benchmarkable.
- **No KV cache.** Generation recomputes the full prefix each
  step. Correctness first; KV caching is an optimization for a
  later saga.
- **No LoRA / quantization.** Those are Saga 15.
- **No embedding visualization or RAG.** Saga 16.

## Quality requirements (every step)

Identical to Sagas 11 / 11.5 / compile-to-rust / 12:

1. TDD: failing test first, then implementation, then refactor.
2. Quality gates must all pass before commit:
   - `cargo test`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo fmt --all` + `cargo fmt --all -- --check`
   - `markdown-checker -f "**/*.md"` if docs touched
   - `sw-checklist`
3. `/mw-cp` checkpoint process.
4. Push immediately after commit.
5. Web UI changes rebuild `pages/` via `scripts/build-pages.sh`.
6. `.agentrail/` changes are committed whenever they change.

## What shipped

### Phase 1 -- LM primitives

- **`embed(vocab_size, d_model, seed)`** is a new `Value::Model`
  variant that holds a `[vocab, d_model]` parameter table.
  `apply(emb, tokens)` gathers one row per integer token id and
  returns `[T, d_model]` (or `[B, T, d_model]`). Gradients flow
  back into the embedding table through `grad(...)`, one-hot
  equivalent verified by gradcheck. Repeated token ids accumulate
  on the matching row.
- **`sinusoidal_encoding(seq_len, d_model)`** is a deterministic
  `[time=seq_len, dim=d_model]` table using the standard
  transformer sinusoid formula. No parameters. Additive pattern:
  `pos_emb = apply(emb, toks) + sinusoidal_encoding(T, d)`.
  Chosen over a learned `positional(max_len, d_model, seed)`
  layer because the demo saw no measurable win from learning and
  the zero-parameter path kept the tutorial surface small.
- **`causal_attention(d_model, heads, seed)`** is the same
  self-attention as `attention(...)` but applies a
  lower-triangular mask (set upper-triangle scores to `-1e9`
  before softmax). Tests verify position 0's output is
  independent of position 1's input, the lower-triangle property
  holds across the full `T x T` grid, and the masked forward is
  still differentiable through the tape. Same four projection
  parameters as `attention` for a given seed.

### Phase 2 -- loss and sampling

- **`cross_entropy(logits, targets)`** returns a scalar mean
  negative log-likelihood. Logits are `[N, V]` (or `[B, T, V]`);
  targets are `[N]` (or `[B, T]`) integer-valued. Max-subtraction
  inside log-softmax keeps the op finite for logits of order
  `1e3`. Fully differentiable wrt logits through the tape. Wrong-
  shape targets surface as `EvalError::ShapeMismatch`.
- **`sample(logits, temperature, seed)`** draws a single integer
  token id from a rank-1 `[V]` logit vector. Temperature 0
  collapses to `argmax`; at high temperature the empirical
  distribution matches softmax over many draws. Determinism on a
  fixed seed is tested end-to-end. **`top_k(logits, k)`** zeroes
  out all but the top-k logits pre-softmax; composed with
  `sample` this is top-k sampling.

### Phase 3 -- end-to-end demos

- **`demos/tiny_lm.mlpl`** ties it all together: load the
  preloaded `tiny_shakespeare_snippet`, `train_bpe` a 280-vocab
  tokenizer, build `[N, T]` next-token pairs with new
  `shift_pairs_x` / `shift_pairs_y` helpers, build a
  `chain(embed, residual(chain(rms_norm, causal_attention)),
  residual(chain(rms_norm, linear, relu_layer, linear)),
  rms_norm, linear)` model, wrap training in `experiment
  "tiny_lm" { train 200 { adam(cross_entropy(...), model, ...);
  loss_metric = cross_entropy(...) } }`, and render the loss
  curve. Loss drops monotonically (within noise) over the 200
  steps.
- **`demos/tiny_lm_generate.mlpl`** trains the same model, then
  runs a 40-token generation loop (`apply` -> `last_row` ->
  `top_k` -> `sample` -> `concat`), decodes the produced ids,
  and renders a `[T, T]` attention heatmap via a new
  `attention_weights(model, X)` builtin that walks the model and
  returns the first attention layer's softmax weights.

### Phase 4 -- tutorials and docs

- Two new web REPL tutorial lessons:
  - **Language Model Basics** -- a pure-forward lesson walking
    from `embed` through `sinusoidal_encoding`,
    `causal_attention`, and `cross_entropy` on a toy 8-vocab,
    4-dim, 4-token example. Runs in well under two seconds in
    the browser because no training happens.
  - **Training and Generating** -- a stripped-down tiny LM wrapped
    in `experiment "tutorial_tiny_lm" { ... }`, a 20-token
    generation loop, and the attention heatmap.
- `docs/milestone-tiny-lm.md` (this file), `docs/status.md`, and
  `docs/saga.md` reflect the saga's shipped state.
- `pages/` was rebuilt via `scripts/build-pages.sh` and committed
  alongside the source so the live demo picks up the new
  lessons and demos on the next Pages deploy.

### Phase 5 -- release

Workspace version bumped to `0.10.0`, tag `v0.10.0` pushed, and
Pages workflow verified. See step 009's commit message for the
full primitive / demo / tutorial summary.

## Planned steps (delivered)

| # | Slug | Phase | What it delivered |
|---|------|-------|-------------------|
| 001 | embedding-layer | 1 | `embed(vocab, d_model, seed)` model + gradcheck |
| 002 | positional-encoding | 1 | `sinusoidal_encoding(T, d)` deterministic table |
| 003 | causal-mask | 1 | `causal_attention(d, heads, seed)` |
| 004 | cross-entropy-loss | 2 | stable `cross_entropy(logits, targets)` |
| 005 | sampling | 2 | `sample(logits, t, seed)` + `top_k(logits, k)` |
| 006 | tiny-lm-train | 3 | `demos/tiny_lm.mlpl` end-to-end training |
| 007 | tiny-lm-generate | 3 | generation loop + `attention_weights` heatmap |
| 008 | tutorials-pages | 4 | two tutorial lessons + docs + `pages/` rebuild |
| 009 | tiny-lm-release-v0100 | 5 | v0.10.0 bump + tag + Pages verification |

Nine steps, same phase split as the original plan.

## Success criteria (met)

- A tiny transformer LM trains end-to-end in a single MLPL
  program on CPU: tokenize -> pairs -> `chain(embed, ...,
  causal_attention, ..., linear)` -> `experiment { train N {
  adam(cross_entropy(apply(...), ...), model, ...) } }` ->
  `loss_curve`.
- The generation loop produces deterministic output on a fixed
  seed and variable output under temperature / top-k changes.
- `attention_weights(model, X)` returns the `[T, T]` softmax
  weights of the first attention layer and renders as an SVG
  heatmap.
- All Saga 11.5 and Saga 12 demos still run unchanged.
- All quality gates green; pages deployed; release tagged.

## Risks and open questions (resolved)

- **Embedding plumbing vs `linear`.** The new `Value::Model`
  embedding variant mirrors `linear`'s existing model-param
  plumbing, so `params(model)`, Adam's per-parameter state
  tracking, and the gradient tape all just work.
- **Causal mask in the lowered tape.** Step 003 applied the
  lower-triangular mask inside the tape-lowered attention path,
  not post-hoc, so gradient flow through the mask remains valid.
- **Cross-entropy numerical stability on untrained logits.**
  Max-subtraction inside log-softmax kept the loss finite for
  logits of order `1e3`, verified in a dedicated test.
- **BPE + training wall-clock on CPU.** The tiny_lm demo's 200
  Adam steps on a 280-vocab BPE tokenizer over
  `tiny_shakespeare_snippet` completes in well under a minute on
  a laptop -- slow enough to matter for iteration, fast enough
  for the demo. MLX (Saga 14) is the proper speedup lever.
- **Generation cost without a KV cache.** Each generated token
  recomputes the full prefix. The tutorial budget (20 tokens,
  short prefix) keeps this interactive; the release demo (40
  tokens) is the upper end for the default config.
