# MLPL User Guide

## Getting Started

### Browser REPL

The fastest way to try MLPL is in the browser:

[https://sw-ml-study.github.io/sw-mlpl/](https://sw-ml-study.github.io/sw-mlpl/)

No installation required. Type expressions and see results instantly.

### Command-Line REPL

```bash
# Build and launch the REPL
cargo run -p mlpl-repl
```

The REPL shows a `mlpl>` prompt. Type expressions and press Enter:

```
mlpl> 1 + 2
3
mlpl> [1, 2, 3] * 10
10 20 30
```

### Running Script Files

Save MLPL code in a `.mlpl` file and run it:

```bash
cargo run -p mlpl-repl -- -f demos/basics.mlpl
```

The REPL executes each line and prints results.

## REPL Commands

| Command | Description |
|---------|-------------|
| `:help` | Show built-in function list and syntax summary |
| `:clear` | Reset all variables to start fresh |
| `:trace on` | Enable execution tracing |
| `:trace off` | Disable execution tracing |
| `:trace` | Show summary of last trace |
| `:trace json` | Print last trace as JSON |
| `:trace json <file>` | Write trace JSON to a file |
| `exit` | Quit the REPL |

## Working with Arrays

### Creating Arrays

```
# Scalars
42
-3.14

# Vectors
[1, 2, 3, 4, 5]

# Matrices (nested arrays)
[[1, 2, 3], [4, 5, 6]]

# Generate a sequence
iota(10)          # 0 1 2 3 4 5 6 7 8 9

# Construct arrays of a specific shape
zeros([3, 4])     # 3x4 matrix of zeros
ones([2, 2])      # 2x2 matrix of ones
fill([5], 3.14)   # vector of five 3.14s

# Seeded random arrays (deterministic for a given seed)
random(42, [3, 4])  # 3x4 uniform [0, 1) values
randn(42, [1000])   # 1000 standard-normal values

# Argmax (flat scalar, or along an axis)
argmax([1, 5, 2, 4])                    # scalar: 1
argmax(reshape(iota(6), [2, 3]), 1)     # per-row argmax

# Synthetic gaussian blobs dataset
# 3 centers, 20 points each -> 60x3 matrix of [x, y, label]
blobs(42, 20, [[0, 0], [3, 3], [-3, 3]])
```

### Reshaping and Transposing

```
x = iota(12)
m = reshape(x, [3, 4])
# 0  1  2  3
# 4  5  6  7
# 8  9  10 11

transpose(m)
# 0 4 8
# 1 5 9
# 2 6 10
# 3 7 11

shape(m)       # 3 4
rank(m)        # 2
```

### Reductions

```
reduce_add([1, 2, 3, 4, 5])          # 15 (sum all)
reduce_mul([1, 2, 3, 4, 5])          # 120 (product all)

m = reshape(iota(6), [2, 3])
reduce_add(m, 0)                      # sum along rows: 3 5 7
reduce_add(m, 1)                      # sum along columns: 3 12
```

## Variables

Variables persist across lines in the same REPL session:

```
x = 42
y = x + 8
y                 # 50

data = [1, 2, 3]
scaled = data * 10
scaled            # 10 20 30
```

Use `:clear` to reset all variables.

## Arithmetic and Broadcasting

All arithmetic operators work element-wise on arrays:

```
[1, 2, 3] + [4, 5, 6]     # 5 7 9
[10, 20] - [3, 7]          # 7 13
[2, 3, 4] * [5, 6, 7]      # 10 18 28
[10, 20, 30] / [2, 4, 5]   # 5 5 6
```

A scalar is broadcast to match the array shape:

```
[1, 2, 3] + 10     # 11 12 13
5 * [1, 2, 3]       # 5 10 15
```

## Linear Algebra

### Dot Product

```
a = [1, 2, 3]
b = [4, 5, 6]
dot(a, b)          # 32 (1*4 + 2*5 + 3*6)
```

### Matrix Multiplication

```
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
matmul(A, B)
# 19 22
# 43 50
```

Matrix-vector multiplication:

```
W = [[1, 2], [3, 4], [5, 6]]
x = reshape([1, 1], [2, 1])
matmul(W, x)
# 3
# 7
# 11
```

## Math Functions

All math functions apply element-wise:

```
exp(1)             # 2.718281828...
log(exp(1))        # 1
sqrt([4, 9, 16])   # 2 3 4
abs([-3, 0, 5])    # 3 0 5
pow([2, 3], [3, 2])  # 8 9
```

## ML Activations

```
sigmoid(0)         # 0.5
sigmoid([0, 1, -1])
# 0.5  0.731...  0.269...

tanh_fn([0, 1, -1])
# 0  0.762...  -0.762...
```

## Comparisons and Statistics

Comparison functions return 0 (false) or 1 (true):

```
gt([3, 1, 4], [2, 2, 2])   # 1 0 1
lt([1, 5, 3], [2, 2, 4])   # 1 0 1
eq([1, 2, 3], [1, 0, 3])   # 1 0 1
```

Compute the mean of an array:

```
mean([2, 4, 6, 8])          # 5
```

## Loops

The `repeat` construct runs a block a fixed number of times:

```
x = 0
repeat 100 { x = x + 1 }
x    # 100
```

Multiple statements in the body:

```
total = 0
count = 0
repeat 5 {
  count = count + 1;
  total = total + count
}
total    # 15
```

## Visualizing Data

The `svg(data, type)` built-in renders an array as an inline SVG
diagram. In the browser REPL the SVG is displayed directly below
the input; in the CLI REPL it prints a one-line summary and can
optionally be written to a file with `--svg-out <dir>`.

### Diagram types

```
# Scatter: Nx2 matrix of (x, y) points
svg([[0,0],[1,1],[2,4],[3,9],[4,16]], "scatter")

# Line: a vector becomes a polyline; Nx2 becomes connected (x,y) points
svg([1, 3, 2, 5, 4, 6], "line")

# Bar: one bar per element of a vector
svg([3, 1, 4, 1, 5, 9, 2, 6], "bar")

# Heatmap: MxN matrix as a colored grid (viridis ramp)
svg(reshape(iota(25), [5, 5]), "heatmap")
```

### Loss curve walkthrough

The `loss_curve.mlpl` demo fits `y = w*x` to a small dataset by
sweeping `w` over a range and computing the mean squared error at
every value:

```
x = [0, 1, 2, 3, 4]
y = [0, 2, 4, 6, 8]              # true slope = 2

ws = iota(25) / 4 - 1            # 25 candidate slopes
WS = reshape(ws, [25, 1])
preds = matmul(WS, reshape(x, [1, 5]))
YS    = matmul(ones([25, 1]), reshape(y, [1, 5]))
errs  = preds - YS
losses = reduce_add(errs * errs, 1) / 5

svg(losses, "line")              # render the loss curve
```

The result is a U-shaped curve with its minimum at `w = 2`.

### Decision boundary

`svg(grid_outputs, "decision_boundary", training)` renders a
classifier's probability surface over a 2D region with the
training points overlaid:

```
gx = grid([0, 1, 0, 1], 20)      # 400 (x, y) points in the unit square
# ... train logistic regression to get w and b ...
gz = matmul(gx, reshape(w, [2, 1])) + b
gp = sigmoid(reshape(gz, [400]))
surface = reshape(gp, [20, 20])
train = [[0,0,0],[0,1,0],[1,0,0],[1,1,1]]
svg(surface, "decision_boundary", train)
```

See `demos/decision_boundary.mlpl` for the full demo.

### Analysis helpers

`svg()` is a low-level primitive. For common diagrams there are
higher-level helpers that compute the right view of the data and
render a complete picture in one call:

```
hist([1, 2, 2, 3, 3, 3, 4, 4, 5], 5)
scatter_labeled([[0,0],[1,1],[0,1],[1,0]], [0, 0, 1, 1])
loss_curve([5.0, 3.0, 2.0, 1.0, 0.5, 0.25])
confusion_matrix([0,1,2,1,0], [0,1,1,1,0])
boundary_2d(grid_probs, [20, 20], training_points, training_labels)
```

`demos/analysis_demo.mlpl` walks through training a classifier and
rendering its loss curve, confusion matrix, and decision boundary
in a single script.

## Execution Tracing

Enable tracing to inspect what MLPL does internally:

```
mlpl> :trace on
Tracing enabled.
mlpl> [1, 2, 3] + [4, 5, 6]
5 7 9

mlpl> :trace
Trace for: [1, 2, 3] + [4, 5, 6]
Events: 3
  [  0] ArrayLit     span=0..9
  [  1] ArrayLit     span=12..21
  [  2] BinOp        span=0..21
```

Export the trace as JSON for external analysis:

```
mlpl> :trace json output.json
Trace written to output.json
```

Run scripts with tracing enabled:

```bash
cargo run -p mlpl-repl -- -f demos/trace_demo.mlpl --trace
```

## Example: Logistic Regression

Train a model to learn the AND gate (output 1 only when both
inputs are 1):

```
# Dataset
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0, 0, 0, 1]

# Initialize weights
w = zeros([2])
b = 0
lr = 1.0
n = 4

# Train for 300 steps
repeat 300 {
  z = matmul(X, reshape(w, [2, 1])) + b;
  pred = sigmoid(z);
  dz = pred - reshape(y, [4, 1]);
  dw = reshape(matmul(transpose(X), dz), [2]) / n;
  db = mean(dz);
  w = w - lr * dw;
  b = b - lr * db
}

# Check predictions
pred = sigmoid(matmul(X, reshape(w, [2, 1])) + b)

# Measure accuracy
rounded = gt(pred, 0.5)
accuracy = mean(eq(reshape(rounded, [4]), y))
accuracy    # 1 (100%)
```

## Labeled Axes

Annotation syntax on assignment attaches axis names as metadata.
Labels propagate through elementwise ops, matmul, reductions, and
activations; a mismatch surfaces as a structured error that names
both shapes:

```
X : [batch, feat] = randn(7, [60, 2])
labels(X)                         # "batch,feat"
reduce_add(X, "feat")             # reduce by axis name
```

See `docs/lang-reference.md` under "Labeled Axes" for `label`,
`relabel`, `reshape_labeled`, and `labels`.

## Autograd, Optimizers, and the Training Loop

Declare a trainable leaf with `param[shape]`. `grad(loss_expr, W)`
lifts the expression onto a reverse-mode tape and returns the
gradient with the same shape as `W`. `adam` / `momentum_sgd` take
either a single param, a list, or a model identifier:

```
W = param[1]
W = randn(1, [1]) * 2             # initialize

train 50 {
  adam(sum(W*W), W, 0.1, 0.9, 0.999, 0.00000001);
  reduce_add(W*W)
}
loss_curve(last_losses)
```

`train N { body }` mirrors `repeat` but also binds the iteration
index to `step` and captures each iteration's final value into a
`last_losses` vector.

## Model DSL

Stack layers as data. `chain(a, b, ...)` composes sequentially;
`residual(block)` adds a skip connection; `apply(m, X)` runs the
forward pass and is differentiable through every owned parameter:

```
mdl = chain(linear(2, 8, 11), tanh_layer(), linear(8, 2, 12))
X : [batch, feat] = matmul(moons(7, 60, 0.08), [[1,0],[0,1],[0,0]])
Y = one_hot(reshape(matmul(moons(7, 60, 0.08), [[0],[0],[1]]), [120]), 2)

train 100 {
  adam(mean((apply(mdl, X) - Y) * (apply(mdl, X) - Y)),
       mdl, 0.05, 0.9, 0.999, 0.00000001);
  mean((apply(mdl, X) - Y) * (apply(mdl, X) - Y))
}
```

Available layers: `linear`, `tanh_layer`, `relu_layer`,
`softmax_layer`, `rms_norm`, `attention`, `causal_attention`,
`embed`, plus `sinusoidal_encoding` for additive positional tables.

## Loading Data

The terminal REPL reads files under a sandbox (`--data-dir <path>`);
the web REPL uses a compiled-in corpus registry instead:

```
# Terminal REPL: cargo run -p mlpl-repl -- --data-dir ./data
text = load("corpus.txt")         # whole-file Value::Str
points = load("points.csv")       # numeric array, header -> labels

# Either REPL:
text = load_preloaded("tiny_corpus")
text = load_preloaded("tiny_shakespeare_snippet")
```

Dataset ops prepare training data without leaving MLPL:

```
data = reshape(iota(12), [6, 2])
s = shuffle(data, 7)
batched = batch(iota(5), 2)        # zero-pads the short tail
mask = batch_mask(iota(5), 2)       # 1 for real rows, 0 for padding
trset = split(iota(10), 0.8, 42)
vaset = val_split(iota(10), 0.8, 42)

for row in reshape(iota(6), [3, 2]) { reduce_add(row) }
last_rows                          # [1, 5, 9]
```

## Tokenizers

Byte-level tokenization is the deterministic baseline; byte-pair
encoding adds a trained merge table on top. Round-trip is lossless
for any UTF-8 input:

```
tokenize_bytes("hello")            # [104, 101, 108, 108, 111]
decode_bytes(tokenize_bytes("round trip"))

bpe = train_bpe("abababab", 260, 7)
apply_tokenizer(bpe, "abababab")
decode(bpe, apply_tokenizer(bpe, "unseen text"))
```

## Experiment Tracking

Wrap a block in `experiment "name" { ... }` to capture every scalar
assigned to a name ending in `_metric` along with the shapes of any
`param` bindings. The terminal REPL additionally writes the record
to `<--exp-dir>/<name>/<timestamp>/run.json`:

```
experiment "baseline" { loss_metric = 0.5; accuracy_metric = 0.82 }
experiment "tweak"    { loss_metric = 0.3; accuracy_metric = 0.91 }
:experiments
compare("baseline", "tweak")
```

## Training a Tiny Language Model

Saga 13 ties everything above together. `embed(V, d, seed)` is a
learned lookup table; `sinusoidal_encoding(T, d)` is deterministic
positional info; `causal_attention` masks the pre-softmax scores so
position `t` cannot peek at `t+1`; `cross_entropy(logits, targets)`
is a numerically-stable fused loss; `sample` + `top_k` plus the
`last_row` / `concat` helpers give you a generation loop:

```
corpus = load_preloaded("tiny_corpus")
tok    = train_bpe(corpus, 260, 0)
ids    = apply_tokenizer(tok, corpus)
X_all  = shift_pairs_x(ids, 8)
Y_all  = shift_pairs_y(ids, 8)
X      = reshape(X_all, [reduce_mul(shape(X_all))])
Y      = reshape(Y_all, [reduce_mul(shape(Y_all))])

V = 260 ; d = 16 ; h = 1
model = chain(embed(V, d, 0),
              causal_attention(d, h, 1),
              rms_norm(d),
              linear(d, V, 2))

experiment "tutorial_tiny_lm" {
  train 30 {
    adam(cross_entropy(apply(model, X), Y),
         model, 0.01, 0.9, 0.999, 0.00000001);
    loss_metric = cross_entropy(apply(model, X), Y)
  }
}
loss_curve(last_losses)

# Generation
prompt = apply_tokenizer(tok, "the ")
seq    = prompt
repeat 20 {
  logits = apply(model, seq);
  last   = last_row(logits);
  nxt    = sample(top_k(last, 20), 0.8, step);
  seq    = concat(seq, nxt)
}
decode(tok, seq)

# Attention heatmap
viz_ids = apply_tokenizer(tok, "the quick")
svg(attention_weights(model, viz_ids), "heatmap")
```

The web REPL's "Training and Generating" tutorial lesson walks
through the same flow interactively. `demos/tiny_lm.mlpl` and
`demos/tiny_lm_generate.mlpl` are the full-size versions (280-vocab
BPE, Shakespeare corpus, residual transformer block).

## Compiling MLPL to a Native Binary

`mlpl-build` takes a `.mlpl` script and produces a self-contained
native binary that only links against `mlpl-rt`. The compiled
program has no interpreter, no parser, and no runtime dispatch --
startup is just the OS loading an executable.

```bash
cargo run -p mlpl-build -- examples/compile-cli/hello.mlpl -o hello
./hello
# -> 42

# Cross-compile the same source to WASM
cargo run -p mlpl-build -- examples/compile-cli/hello.mlpl \
    --target wasm32-unknown-unknown -o hello.wasm
```

See `examples/compile-cli/README.md` for a complete walkthrough,
and `docs/compiling-mlpl.md` for the three-way comparison of the
interpreter, the `mlpl!` proc macro, and the `mlpl build` path.

## Demo Scripts

The `demos/` directory contains ready-to-run examples:

```bash
cargo run -p mlpl-repl -- -f demos/basics.mlpl               # arithmetic, arrays, variables
cargo run -p mlpl-repl -- -f demos/matrix_ops.mlpl            # reshape, transpose, reductions
cargo run -p mlpl-repl -- -f demos/computation.mlpl           # multi-step computation
cargo run -p mlpl-repl -- -f demos/repeat_demo.mlpl           # loop construct
cargo run -p mlpl-repl -- -f demos/logistic_regression.mlpl   # ML training
cargo run -p mlpl-repl -- -f demos/loss_curve.mlpl            # SVG loss curve
cargo run -p mlpl-repl -- -f demos/decision_boundary.mlpl     # 2D classifier
cargo run -p mlpl-repl -- -f demos/analysis_demo.mlpl         # analysis helpers
cargo run -p mlpl-repl -- -f demos/kmeans.mlpl                # K-Means clustering
cargo run -p mlpl-repl -- -f demos/pca.mlpl                   # PCA via power iteration
cargo run -p mlpl-repl -- -f demos/softmax_classifier.mlpl    # 3-class softmax
cargo run -p mlpl-repl -- -f demos/tiny_mlp.mlpl              # 2-8-2 MLP on XOR-like data
cargo run -p mlpl-repl -- -f demos/moons_mlp.mlpl             # chain + train + adam on moons
cargo run -p mlpl-repl -- -f demos/circles_mlp.mlpl           # same, on circles
cargo run -p mlpl-repl -- -f demos/attention.mlpl             # Q K^T / sqrt(d) pattern
cargo run -p mlpl-repl -- -f demos/transformer_block.mlpl     # residual attention + MLP
cargo run -p mlpl-repl -- -f demos/tiny_lm.mlpl               # tiny LM training (Saga 13)
cargo run -p mlpl-repl -- -f demos/tiny_lm_generate.mlpl      # training + generation + attention heatmap
cargo run -p mlpl-repl -- -f demos/trace_demo.mlpl --trace    # execution tracing
```
