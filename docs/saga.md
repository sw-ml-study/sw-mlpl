# Saga

## Saga -1: Repo scaffolding (COMPLETE)
Repo compartmentalization scaffolding.

## Saga 0: Foundation (COMPLETE)
Foundation and contracts.

## Saga 1: Dense tensor substrate v1 (COMPLETE)
Shape, DenseArray, reshape, transpose, indexing.

## Saga 2: Parser and evaluator foundation (COMPLETE)
Lexer, AST, parser with precedence, AST-walking evaluator.

## Saga 3: CLI and REPL v1 (COMPLETE)
Working REPL with all v1 syntax, built-in functions, tracing.

## Saga 4: Structured trace v1 (COMPLETE)
TraceEvent/Trace types, evaluator instrumentation, JSON export.

## Saga 5: Visual web viewer v1 (DEFERRED)
Deferred to post-MVP. MVP uses CLI + JSON trace export.

## MVP
Sagas 0 through 4 complete. MVP ships with REPL + trace export.

## Saga 6: ML foundations (COMPLETE)
Sigmoid, tanh_fn, pow, comparison operators, axis reductions, mean,
array constructors (zeros, ones, fill), and the first end-to-end
logistic regression demo. Delivered v0.2.

## Saga 7: SVG visualization v1 (COMPLETE)
`mlpl-viz` crate, `svg(data, type[, aux])` built-in, diagram types
(scatter, line, bar, heatmap, decision_boundary), `grid()` helper,
high-level analysis helpers (hist, scatter_labeled, loss_curve,
confusion_matrix, boundary_2d), browser REPL inline SVG rendering,
and download button. Delivered v0.3.

## Saga 11: Model DSL (COMPLETE)
Composition primitives for neural-net models: a new `Value::Model`
runtime value, atomic `linear(in, out, seed)` layers,
parameter-free activation layers (`tanh_layer`, `relu_layer`,
`softmax_layer`), sequential `chain(...)` composition,
`residual(block)` skip connections, `rms_norm(dim)` normalization,
and `attention(d_model, heads, seed)` multi-head self-attention
(tape-lowered for `heads=1`, forward-only for `heads>1`). A
`params(model)` walker returns the flat parameter list, and
optimizers now accept a model identifier directly so
`adam(loss, mdl, lr, b1, b2, eps)` trains every weight the model
owns through differentiable `apply(mdl, X)`. The v0.6 `moons_mlp`
and `tiny_mlp` demos were rewritten as one-line `chain(...)`
expressions, and a new `transformer_block.mlpl` demo stacks
`residual(chain(rms_norm, attention))` and `residual(chain(
rms_norm, linear, relu_layer, linear))` twice to train a tiny
2-layer transformer block end-to-end (loss 143.87 -> 1.02 over
100 Adam steps, strictly monotonic). New REPL introspection
commands -- `:vars`, `:models`, `:fns`, `:wsid`, `:describe` --
ship in both `mlpl-repl` and `mlpl-web`, and a new "Model
Composition" tutorial lesson walks from a single `linear` through
a chain MLP to Adam inside `train { }` on the moons dataset.
Delivered v0.7. See `docs/milestone-modeldsl.md`.

## Saga 10: Optimizers + training loop (COMPLETE)
Built proper training infrastructure on top of Saga 9 autograd.
New built-ins: `momentum_sgd(loss, params, lr, beta)` and
`adam(loss, params, lr, b1, b2, eps)` with per-parameter state
held in an `OptimizerState` map on the evaluation environment;
`cosine_schedule(step, total, lr_min, lr_max)` and
`linear_warmup(step, warmup, lr)` pure scalar schedules; and
two non-linear synthetic datasets `moons(seed, n, noise)` and
`circles(seed, n, noise)` returning `[N, 3]` `[x, y, label]`
matrices in the same layout as `blobs`. New `train N { body }`
language construct binds the iteration index to `step` inside
the body and captures each iteration's final value into a
`last_losses` 1-D array, replacing the manual
`repeat { grad; manual update; record loss }` recipe. Two new
demos -- `demos/moons_mlp.mlpl` and `demos/circles_mlp.mlpl` --
train a tanh MLP with `adam` inside `train { }` and render the
decision boundary with `boundary_2d`. New "Optimizers and
Schedules" tutorial lesson added to the web REPL. Delivered v0.6.

## Saga 9: Autograd v1 (COMPLETE)
Reverse-mode autograd as a language primitive. New `mlpl-autograd`
crate provides a tape-based `Tensor` with backward over add, sub,
mul, div, neg, exp, log, relu, tanh, sigmoid, softmax, sum, mean,
transpose, reshape, and matmul, all gradcheck-verified against
finite differences. Parser surface adds `param[shape]` and
`tensor[shape]` constructors, and a new `grad(expr, wrt)` built-in
lifts array expressions onto the tape and returns the gradient
with respect to a tracked parameter. The v0.4 `tiny_mlp` and
`softmax_classifier` demos were ported to use `param + grad`
instead of hand-written backprop, and a new "Automatic
Differentiation" tutorial lesson walks from a scalar minimization
to a one-layer linear regression. Delivered v0.5.

## Saga 8: ML demos (COMPLETE)
Synthetic data primitives (random, randn, argmax, blobs) and
higher-level ML built-ins (softmax, one_hot), plus six demos wired
into the browser REPL: k-means clustering, PCA via power iteration,
a linear softmax classifier, a tiny MLP on XOR-style data, and a
scaled dot-product attention pattern. Tutorial lessons added for
each. Delivered v0.4.

## Saga 11.5: Named Axes and shape introspection (COMPLETE)
Axis-labeled shapes threaded through `Value::Array` via a new
`LabeledShape` type on `mlpl-core`. `label(x, [...])` and `relabel(x,
[...])` primitives, annotation syntax on assignment (`x : [batch,
time, dim] = ...`), and `reshape_labeled(x, dims, labels)` as the
opt-in re-labeling path. Label propagation through elementwise
ops (with one-None-one-Some accepted and mismatches rejected),
matmul (contraction axis validated, outer dims passed through),
reduce/argmax (reduced axis's label dropped), and `map()`
(preserves labels through every math builtin). `reduce_add`,
`reduce_mul`, `argmax`, and `softmax` accept an axis name string
in place of an integer. Structured `EvalError::ShapeMismatch
{ op, expected: LabeledShape, actual: LabeledShape }` at the
evaluator boundary, with Display rendering as `op: expected
[seq=N, d=M], got [time=N, d=M]`. Label-aware `:vars` and
`:describe` (using `LabeledShape` Display), and trace JSON that
round-trips axis labels for labeled arrays and omits the key
entirely for unlabeled ones. New "Named Axes" tutorial lesson in
the web REPL; "Model Composition" lesson now annotates X as
`[batch, feat]` so labels flow through `apply(mdl, X)`. Delivered
v0.7.5. See `docs/milestone-named-axes.md`.

## Saga: Compile-to-Rust (COMPLETE)
Lowers MLPL source to Rust `TokenStream` shared by three targets:
the `mlpl!` proc macro (compile-time embed inside Rust apps), the
`mlpl build foo.mlpl -o bin` subcommand (native-binary and
cross-compile via cargo+rustc, verified for native and
wasm32-unknown-unknown), and a leaner future WASM path that
reuses the same pipeline. New crates: `mlpl-rt` (runtime target;
typed primitive wrappers around `DenseArray`/`LabeledShape`),
`mlpl-lower-rs` (AST -> `TokenStream` with path-configurable
runtime prefix and static matmul contraction checks when both
operands' labels are known at lower time), `mlpl-macro`
(proc-macro wrapper), `mlpl` (user-facing facade with hidden
`__rt` re-export), `mlpl-parity-tests` (parity harness -- nine
curated programs run through both paths and agreement is
asserted bit-for-bit). Static label mismatches surface as
`LowerError::StaticShapeMismatch` which the proc macro converts
to `compile_error!`; the `mlpl-build` CLI surfaces them as
`mlpl-build: ...` prefixed stderr before cargo is even invoked.
Measured 9.05x speedup on a 100x100 reshape+reduce workload
(interpreter 479us -> compiled 53us, median of 5). Deferred:
`TensorCtor` (`param`/`tensor`), `Repeat`, `Train`, autograd
(`grad`), optimizers (`adam`/`momentum_sgd`), and Model DSL
(`chain`/`linear`/activations) are all out of compile scope for
this saga; they require tape-state or loop lowering. Delivered
v0.8. See `docs/milestone-compile-to-rust.md`.

## Saga 13: Tiny language model end-to-end (IN PROGRESS)
First saga that proves the platform thesis: a few lines of MLPL
train a small transformer LM, generate text from a prompt, and
render the attention pattern -- all on CPU, all reproducible via
experiment tracking. Six new primitives filled the gap between
Saga 12's tokenizer surface and a working LM: `embed(vocab,
d_model, seed)` as a `Value::Model` with a learned `[vocab,
d_model]` table and gradient flow; deterministic
`sinusoidal_encoding(T, d_model)` additive positional tables;
`causal_attention(d_model, heads, seed)` with a lower-triangular
pre-softmax mask so position `t` cannot peek at `t+1`;
numerically-stable `cross_entropy(logits, targets)` over integer
targets with max-subtraction log-softmax; `sample(logits, t,
seed)` multinomial draws plus `top_k(logits, k)` pre-softmax
restriction; and a `last_row` / `concat` / `attention_weights`
triple that makes a generation loop and a `[T, T]` attention
heatmap one-liner. New demos `demos/tiny_lm.mlpl` (end-to-end
training) and `demos/tiny_lm_generate.mlpl` (generation +
heatmap) wire every Saga 13 primitive together with Saga 12's
BPE tokenizer, `shift_pairs_x`/`shift_pairs_y` next-token pair
windows, and `experiment "name" { }`-tracked `train N { adam(... )
}`. Two new web REPL tutorial lessons -- "Language Model Basics"
(forward-only, runs in <2s) and "Training and Generating"
(stripped-down training loop + 20-token generation + attention
heatmap) -- ship alongside. Target v0.10.0; step 009 cuts the
release tag. See `docs/milestone-tiny-lm.md`.

## Saga 12: Tokenizers, datasets, and experiment tracking (COMPLETE)
Closes the last surface-only gap before the Tiny LM. File IO:
`load("rel.csv")` / `load("rel.txt")` reads through an
`Environment::data_dir` sandbox (terminal REPL `--data-dir`
flag); `load_preloaded("name")` serves compiled-in corpora for
the web REPL. Dataset prep: `shuffle(x, seed)` Fisher-Yates row
permutation, `batch(x, size)` with zero-padded short batches +
matching `batch_mask`, `split(x, frac, seed)` and
`val_split(x, frac, seed)` returning disjoint chunks; `for row
in ds { body }` streaming iteration with `last_rows` capture
(new `Token::For`/`Token::In` keywords, `Expr::For` AST
variant). Byte-level BPE: `tokenize_bytes` / `decode_bytes`
primitives; `Value::Tokenizer` runtime variant with
`TokenizerSpec::ByteLevel` and `TokenizerSpec::BpeMerges`;
`train_bpe(corpus, vocab_size, seed)` with deterministic
lex-smallest tie-breaking; `apply_tokenizer(tok, text)` +
`decode(tok, tokens)` byte-lossless round-trip for any UTF-8
input. Experiment tracking: `experiment "name" { body }` scoped
form captures `_metric`-suffixed scalars and bound params'
shapes into `ExperimentRecord`s, logged to
`env.experiment_log` and (terminal REPL `--exp-dir`)
`<dir>/<name>/<ts>/run.json` via serde_json; `:experiments`
REPL command merges memory + disk; `compare(a, b)` builtin
returns side-by-side per-metric deltas. Byproduct: lexer UTF-8
fix so string literals now carry arbitrary Unicode code points
(previously was `b as char` Latin-1 mojibake). Three new
tutorial lessons in the web REPL ("Loading Data", "Tokenizing
Text", "Experiments"). String-valued variable bindings
(`corpus = load_preloaded("...")`) via new
`Environment::strings` map. Delivered v0.9.0. See
`docs/milestone-tokenizers.md`.
