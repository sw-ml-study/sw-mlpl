# Tiny Language Model End-to-End Milestone (Saga 13, v0.10.0)

## Why this exists

Saga 13 is the first saga that proves the MLPL platform thesis
end-to-end: a user writes a handful of lines of MLPL, and a
small transformer-based language model trains, generates
text, and visualizes its own attention -- all on CPU, with
reproducible experiment tracking.

Every prior saga has been building toward this. Saga 9 gave
us reverse-mode autograd; Saga 10 gave us optimizers and the
`train { }` loop; Saga 11 gave us `chain` / `residual` /
`attention` / `rms_norm` for composing models; Saga 11.5
added shape checking via labeled axes; Saga 12 added file IO,
batched dataset iteration, a byte-level BPE tokenizer, and
`experiment "name" { }` for tracking runs. What is still
missing for a real LM:

1. **Token embeddings.** `Value::Model` has `linear(in,out)`
   but no `embed(vocab, d_model)`. Language models start with
   a learned lookup from integer token ids to vectors.
2. **Positional information.** `attention` today is
   permutation-invariant over the sequence axis. An LM needs
   positional encoding (sinusoidal or learned) added to
   embeddings.
3. **Causal masking.** Existing `attention` attends to all
   positions. An LM needs a lower-triangular causal mask so
   position `t` cannot peek at `t+1`.
4. **Cross-entropy loss.** Autograd knows `sum`, `log`,
   `softmax`, `exp`, but there is no fused `cross_entropy`
   over integer targets. We need a numerically-stable
   log-softmax + NLL primitive that works on `[B*T, V]` logits
   and `[B*T]` integer targets.
5. **Sampling.** `argmax` exists; multinomial sampling over a
   temperature-scaled softmax with optional top-k does not.
6. **Generation loop.** A small helper or tutorial pattern
   that, given a trained model and a prompt, iteratively
   samples the next token and appends it.

Saga 13 fills those gaps, then demos the whole stack with a
tiny character- or BPE-level LM trained on a small corpus,
with attention-map and loss-curve visualization.

## Non-goals

- **No new backends.** CPU only. MLX / CUDA are Sagas 14, 17.
- **No pretraining claims.** Target is ~1-5M params on a
  ~100KB corpus -- enough to see loss drop and coherent-ish
  snippets, not benchmarkable.
- **No KV cache.** Generation recomputes the full prefix each
  step. KV caching is an optimization; correctness first.
- **No LoRA / quantization.** Those are Saga 15.
- **No embedding visualization / RAG.** Saga 16.

## Quality requirements (every step)

Identical to Sagas 11 / 11.5 / compile-to-rust / 12:

1. TDD: failing test first, then implementation, then refactor.
2. Quality gates must all pass before commit:
   - `cargo test`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo fmt --all` + `cargo fmt --all -- --check`
   - `markdown-checker -f "**/*.md"` if docs touched
   - `sw-checklist`
3. Use `/mw-cp` checkpoint process.
4. Push immediately after commit.
5. Web UI changes rebuild `pages/` via `scripts/build-pages.sh`.
6. `.agentrail/` changes are committed whenever they change.

## What already exists

- `param[shape]` / `tensor[shape]` constructors, `grad(expr, wrt)`.
- `adam` / `momentum_sgd` optimizers with per-parameter state;
  `train N { body }` with implicit `step` and `last_losses`.
- Model DSL: `linear(in, out, seed)`, `chain(...)`,
  `residual(block)`, `rms_norm(dim)`, `attention(d, heads, seed)`
  (heads=1 tape-lowered), `apply(mdl, X)`, `params(model)`.
- Labeled axes: `[batch, time, dim]` annotations propagate
  through elementwise / matmul / reduce; `axis="time"` in
  reductions.
- Saga 12 data surface: `load("...")`, `load_preloaded(...)`,
  `shuffle` / `batch` / `batch_mask` / `split` / `val_split`,
  `for row in ds { ... }` / `last_rows`.
- Byte-level + BPE tokenizer: `tokenize_bytes` / `decode_bytes`,
  `train_bpe(corpus, vocab, seed)`, `apply_tokenizer(tok, text)`,
  `decode(tok, tokens)`; `Value::Tokenizer` runtime variant.
- Experiment tracking: `experiment "name" { body }` captures
  `_metric`-suffixed scalars; `:experiments` REPL command;
  `compare(a, b)` returns per-metric deltas; on-disk run.json
  via `--exp-dir`.
- Softmax, log, exp, sum, mean, matmul, reshape all
  differentiable through the tape.

## Phase 1 -- LM primitives (3 steps)

### Step 001 -- embedding layer
New `Value::Model` variant for token embeddings. Builtin
`embed(vocab_size, d_model, seed)` returns a model with a
`[vocab, d_model]` parameter table. `apply(emb, tokens)` where
`tokens` is a `[B, T]` integer array (or `[B*T]`) returns a
`[B, T, d_model]` float array, with gradient flowing back into
the embedding table through `grad(...)`. Gradcheck against
finite differences on a small `(vocab=5, d=3)` table.

### Step 002 -- positional encoding
Additive positional information. Pick one of:
(a) `positional(max_len, d_model, seed)` learned layer, or
(b) `sinusoidal_encoding(seq_len, d_model)` deterministic
    table (no params).
Both return a `[T, d_model]` (or `[1, T, d_model]`
broadcastable) array that gets added to embeddings. Tests:
deterministic output for (b); gradcheck + `params()` walk for
(a). Decision between (a) and (b) during TDD -- start with
(b) for simplicity, add (a) only if the demo benefits.

### Step 003 -- causal mask for attention
Existing `attention` attends to all positions. Add causal
masking. Either:
(a) new `causal_attention(d_model, heads, seed)` builtin, or
(b) optional third arg / flag to `attention` that toggles
    causal masking.
Implementation: lower-triangular mask applied to pre-softmax
scores (set upper triangle to a large negative number).
Tests: a [B=1, T=3, d=2] forward pass with causal attention
must give position-0 output that depends only on position 0's
input (verified by perturbing position 1 and checking
position-0 output is unchanged). Gradcheck preserved.

## Phase 2 -- loss + sampling (2 steps)

### Step 004 -- cross-entropy loss
`cross_entropy(logits, targets)` builtin. `logits` is
`[N, V]` float, `targets` is `[N]` integer. Returns a scalar
`-mean(log_softmax(logits)[i, targets[i]])`. Numerically
stable via max-subtraction inside log-softmax. Fully
differentiable through the tape wrt `logits`. Tests:
(i) matches a hand-written `mean(-log(softmax(x)[target]))`
pipeline on a `[4, 3]` example; (ii) gradcheck against finite
differences; (iii) targets of the wrong shape produce a clear
`EvalError::ShapeMismatch`.

### Step 005 -- sampling
`sample(logits, temperature, seed)` builtin for multinomial
sampling. `logits` is a 1-D `[V]` float array; returns a
scalar integer token id. Temperature = 0 collapses to
`argmax`. Add `top_k(logits, k)` that zeroes out all but the
top-k logits (pre-softmax). Compose for top-k sampling as
`sample(top_k(logits, k), t, seed)`. Tests: determinism on
fixed seed; temperature=0 matches `argmax`; top-k=1 matches
argmax regardless of temperature; distribution approximates
softmax at high temperature over many draws.

## Phase 3 -- end-to-end (2 steps)

### Step 006 -- tiny LM demo (training)
`demos/tiny_lm.mlpl`. End-to-end:
1. `corpus = load_preloaded("tiny_shakespeare_snippet")` (or
   similarly small, ~50-100KB preloaded text).
2. `tok = train_bpe(corpus, vocab_size=256, seed=0)`.
3. `ids = apply_tokenizer(tok, corpus)`.
4. `(X, Y) = shift_pairs(ids, block_size=32)` -- helper to
   build next-token prediction pairs; added if not already
   present as part of step 006.
5. `model = chain(embed(V, d_model), add_positional(T),
   residual(chain(rms_norm(d), causal_attention(d, h))),
   residual(chain(rms_norm(d), linear(d, 4d), relu_layer,
   linear(4d, d))), rms_norm(d), linear(d, V))`.
6. `experiment "tiny_lm" { train N { loss = cross_entropy(
   apply(model, X_batch), Y_batch); adam(loss, model,
   lr, ...); loss_metric = loss } }`.
7. `loss_curve(last_losses)` inline SVG.
Measurable outcome: loss drops monotonically (or near) by
>=40% over N steps. Wire into the web REPL demo list.

### Step 007 -- generation + attention viz
`demos/tiny_lm_generate.mlpl`. Given a trained `model` from
step 006 (either loaded from checkpoint if we've added that,
or trained inline with a tiny budget), implement a sampling
loop: given a prompt string, tokenize, then iteratively
`apply` the model, take the last position's logits, `sample`,
append, decode. Render generated text via the REPL's string
output path. Add an attention-map heatmap: extract one
forward pass's per-head attention weights and visualize as a
`heatmap` SVG. Decision during TDD: either reuse a hook on
`attention` to capture weights, or add `attention_weights(
model, X)` as a separate read-only builtin.

## Phase 4 -- docs (1 step)

### Step 008 -- tutorial lessons + pages rebuild
Add two new web REPL tutorial lessons:
- **"Language Model Basics"** -- walks from `embed` +
  positional + causal attention to a forward pass and
  `cross_entropy`.
- **"Training and Generating"** -- runs the tiny LM training
  loop (smaller budget) and the generation loop, showing the
  attention heatmap.
Update `docs/status.md` one-liner, add
`docs/milestone-tiny-lm.md` retrospective, rebuild
`pages/` via `scripts/build-pages.sh`, commit both the source
and the built artifact in the same commit.

## Phase 5 -- release (1 step)

### Step 009 -- release v0.10.0
Bump workspace version to `0.10.0`. Write release commit
message summarizing primitives (`embed`,
`sinusoidal_encoding`/`positional`, causal attention,
`cross_entropy`, `sample`/`top_k`), demos (`tiny_lm.mlpl`,
`tiny_lm_generate.mlpl`), and tutorials. Tag `v0.10.0`. Push
commit and tag. Verify pages workflow deploys the updated
demo list.

## Dependencies and risk

- Step 001 (embedding) is a new `Value::Model` variant -- the
  main risk is integrating with `params(model)` and the Adam
  state map cleanly. Mitigation: mirror `linear`'s existing
  model-param plumbing.
- Step 003 (causal mask) touches the existing `attention`
  builtin which is tape-lowered for heads=1. Risk: the mask
  must be applied in the lowered tape path, not post-hoc.
- Step 004 (cross-entropy) risk: numerical stability on
  logits drawn from an untrained `[B*T, 256]` output. Always
  subtract max inside log-softmax.
- Step 006 (demo) risk: a 256-vocab BPE on a 100KB corpus may
  train very slowly on pure-interpreter CPU. Accept it. If
  unworkable, fall back to a smaller vocab + shorter block
  size; MLX (Saga 14) is the proper speedup lever, not this
  saga.
- Step 007 generation risk: no KV cache means O(T^2) per
  generated token. Keep T small in the demo (~64).
