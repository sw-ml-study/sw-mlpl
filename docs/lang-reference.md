# MLPL Language Reference

## Numeric Literals

```
42        # integer
-3        # negative integer
1.5       # float
-0.25     # negative float
```

Integers are sequences of digits. Floats contain a decimal point with
digits on both sides. No scientific notation.

## String Literals

```
"scatter"
"line"
"hello"
```

Double-quoted ASCII strings. Strings are a separate value kind from
arrays and are used as type-name arguments to built-ins like `svg()`.
Strings cannot be combined with numeric operators.

## Identifiers

```
x
my_var
result2
_temp
```

Start with an ASCII letter or underscore. Contain ASCII letters,
digits, and underscores. Case-sensitive.

## Array Literals

```
[1, 2, 3]              # vector (rank 1)
[[1, 2], [3, 4]]       # matrix (rank 2)
[[1, 2, 3], [4, 5, 6]] # 2x3 matrix
```

Square brackets with comma-separated elements. Nesting creates
higher-rank arrays. All inner arrays must have the same length.
A bare number (no brackets) is a scalar (rank 0).

## Operators

Binary infix operators for element-wise arithmetic:

| Operator | Description |
|----------|-------------|
| `+` | Add |
| `-` | Subtract |
| `*` | Multiply |
| `/` | Divide |
| `-x` | Unary negation |

Precedence (high to low):

1. Unary `-` (prefix negation)
2. `*`, `/` (left-associative)
3. `+`, `-` (left-associative)
4. `=` (assignment, right-associative)

Parentheses override precedence: `(x + y) * z`

Operators apply element-wise on arrays of the same shape, or
broadcast a scalar to match the other operand's shape:

```
[1, 2, 3] + [4, 5, 6]    # [5, 7, 9]
[1, 2, 3] * 10            # [10, 20, 30]
```

## Assignment

```
x = [1, 2, 3]
y = reshape(x, [1, 3])
z = x + 1
```

Single `=` binds a name to a value. No `let` keyword. Reassignment
is allowed. Scope is flat. Assignment returns the assigned value.

## Function Calls

```
reshape([1, 2, 3, 4], [2, 2])
iota(6)
reduce_add([1, 2, 3])
matmul(A, B)
```

Named function followed by parenthesized, comma-separated arguments.

## Loop Construct

```
repeat N { body }
```

Execute the body N times. N must evaluate to a non-negative integer.
The body can contain multiple statements separated by semicolons or
newlines. Returns the result of the last expression in the final
iteration (or scalar 0 if N is 0).

```
x = 0
repeat 10 { x = x + 1 }
# x is now 10
```

## Training Loop

```
train N { body }
```

Like `repeat`, but additionally binds the iteration index to `step`
inside the body and captures each iteration's final expression value
into a `last_losses` 1-D array in the environment. Use with
`momentum_sgd` / `adam` for training loops:

```
train 100 {
  adam(loss_expr, model, 0.01, 0.9, 0.999, 0.00000001);
  loss_expr
}
loss_curve(last_losses)
```

## For-Row Iteration

```
for ident in expr { body }
```

Iterates `ident` over each rank-(r-1) slice of a rank-r array along
axis 0. The body runs once per slice; each iteration's final value
is collected into a `last_rows` vector in the environment (mirrors
`train`'s `last_losses`). Use for streaming over a dataset when a
full batched representation doesn't fit:

```
for row in reshape(iota(6), [3, 2]) { reduce_add(row) }
last_rows   # [1, 5, 9]
```

## Experiment Block

```
experiment "name" { body }
```

A scoped form that records every scalar assigned to a name ending
in `_metric` during the body, along with the shapes of any `param`
bindings. The record lands in the REPL's in-memory experiment log
(web REPL) and also on disk under `--exp-dir/<name>/<timestamp>/run.json`
when the terminal REPL is invoked with `--exp-dir`.

```
experiment "baseline" {
  train 50 { adam(loss, model, 0.01, 0.9, 0.999, 0.00000001); loss };
  loss_metric = loss
}
compare("baseline", "variant")
:experiments
```

## Labeled Axes

An assignment may carry axis labels as metadata so label mismatches
in downstream ops surface with a `ShapeMismatch` error that names
both labeled shapes. Labels propagate through elementwise ops,
matmul (contraction axis validated), reductions (the reduced axis's
label drops), and `map()`.

```
M : [batch, feat] = reshape(iota(6), [2, 3])
labels(M)                         # "batch,feat"
reduce_add(M, "feat")             # reduce by axis name
labels(transpose(M))              # swaps labels alongside dims
```

See `label(x, [...])`, `relabel(x, [...])`, and
`reshape_labeled(x, dims, labels)` in the built-ins table.

## Parameters and Autograd

```
W = param[2, 3]        # trainable leaf; tape-tracked
T = tensor[2, 3]       # non-trainable tape tensor
grad(loss_expr, W)     # gradient of a scalar wrt a param
```

`param[shape]` declares a zero-initialized trainable leaf
(typically immediately overwritten with `randn(seed, shape) * scale`).
`tensor[shape]` declares an ordinary tape-tracked tensor. `grad`
lifts an array expression onto the reverse-mode tape and returns
the gradient with the same shape as the `wrt` operand.

## Comments

```
# this is a comment
x = 42  # inline comment
```

`#` starts a line comment that runs to end of line.

## Statement Separation

```
x = 1; y = 2        # semicolon separates
x = 1
y = 2                # newline separates
```

Newlines and semicolons are both statement separators.

## Built-in Functions

### Array Operations

| Function | Args | Description |
|----------|------|-------------|
| `iota(n)` | 1 | Integers 0, 1, ..., n-1 as a vector |
| `shape(a)` | 1 | Dimension vector of array |
| `rank(a)` | 1 | Number of dimensions (scalar) |
| `reshape(a, dims)` | 2 | Reshape array to new dimensions |
| `transpose(a)` | 1 | Reverse axis order |
| `reduce_add(a)` | 1 | Sum all elements |
| `reduce_add(a, axis)` | 2 | Sum along a specific axis |
| `reduce_mul(a)` | 1 | Product of all elements |
| `reduce_mul(a, axis)` | 2 | Product along a specific axis |

### Linear Algebra

| Function | Args | Description |
|----------|------|-------------|
| `dot(a, b)` | 2 | Dot product of two vectors |
| `matmul(a, b)` | 2 | Matrix multiplication |

### Math Functions

| Function | Args | Description |
|----------|------|-------------|
| `exp(a)` | 1 | Element-wise e^x |
| `log(a)` | 1 | Element-wise natural logarithm |
| `sqrt(a)` | 1 | Element-wise square root |
| `abs(a)` | 1 | Element-wise absolute value |
| `pow(a, b)` | 2 | Element-wise a^b |

### ML Activations

| Function | Args | Description |
|----------|------|-------------|
| `sigmoid(a)` | 1 | Logistic sigmoid: 1 / (1 + exp(-x)) |
| `tanh_fn(a)` | 1 | Hyperbolic tangent |

### Comparison and Statistics

| Function | Args | Description |
|----------|------|-------------|
| `gt(a, b)` | 2 | Element-wise greater-than (returns 0 or 1) |
| `lt(a, b)` | 2 | Element-wise less-than (returns 0 or 1) |
| `eq(a, b)` | 2 | Element-wise equality (returns 0 or 1) |
| `mean(a)` | 1 | Arithmetic mean of all elements |

### Array Constructors

| Function | Args | Description |
|----------|------|-------------|
| `zeros(shape)` | 1 | Array of zeros with given shape |
| `ones(shape)` | 1 | Array of ones with given shape |
| `fill(shape, v)` | 2 | Array filled with value v |
| `grid(bounds, n)` | 2 | n*n by 2 matrix of (x, y) points over [xmin,xmax] x [ymin,ymax]; bounds is [xmin, xmax, ymin, ymax] |
| `random(seed, shape)` | 2 | Seeded uniform [0, 1) array with the given shape. `seed` is a scalar integer; `shape` is a vector of dimensions. Deterministic for a given seed. |
| `randn(seed, shape)` | 2 | Seeded standard-normal array (mean 0, variance 1), same shape semantics as `random`. Implemented via Box-Muller on the same xorshift64 stream. |
| `argmax(a)` | 1 | Scalar index of the maximum element over all elements of `a` (flat). |
| `argmax(a, axis)` | 2 | Index (as f64) of the max along `axis`; output rank is one less than input. Ties go to the first occurrence. |
| `blobs(seed, n_per_class, centers)` | 3 | Seeded 2D gaussian-blob dataset. `centers` is a Kx2 matrix (or length-2K vector) of cluster centers; returns an Nx3 matrix where each row is `[x, y, label]`, with `N = K * n_per_class` and noise sigma 0.15. |
| `softmax(a, axis)` | 2 | Softmax along `axis`, stabilized by subtracting the per-group max before exponentiation. |
| `one_hot(labels, k)` | 2 | Convert a length-N label vector to an `NxK` one-hot matrix. |
| `cross_entropy(logits, targets)` | 2 | Scalar mean negative log-likelihood. `logits` is `[N, V]` or `[B, T, V]`; `targets` is `[N]` or `[B, T]` integer-valued. Fused, numerically-stable log-softmax + NLL; fully differentiable wrt `logits` via `grad(...)`. |
| `sample(logits, temperature, seed)` | 3 | Categorical sample from a 1-D `[V]` logit vector. Returns a scalar integer token id. `temperature == 0.0` collapses to `argmax(logits)`; otherwise draws from `softmax(logits / temperature)` via inverse-CDF on a single seeded uniform. Same `(logits, temperature, seed)` always yields the same id. |
| `top_k(logits, k)` | 2 | Return a `[V]` logit vector with all but the top-`k` entries replaced by `-inf`. Pure (no randomness). Compose with `sample` for top-k sampling: `sample(top_k(logits, k), temperature, seed)`. |
| `moons(seed, n_per_class, noise)` | 3 | Seeded two-moons dataset; returns an `Nx3` matrix of `[x, y, label]` for `N = 2 * n_per_class`. |
| `circles(seed, n_per_class, noise)` | 3 | Seeded two-concentric-circles dataset, same `Nx3` layout as `moons`. |

### Labeled Axes

| Function | Args | Description |
|----------|------|-------------|
| `label(x, names)` | 2 | Attach axis labels to an array. `names` is a rank-1 string array; length must equal the rank of `x`. Use `""` for "no label" on a single axis. |
| `relabel(x, names)` | 2 | Like `label`, but explicitly overrides any existing labels on `x`. |
| `reshape_labeled(x, dims, names)` | 3 | Combine `reshape` and `label` in one call. New axes get the given names; plain `reshape` clears labels. |
| `labels(x)` | 1 | Return the axis labels of `x` as a comma-joined string ("" for unlabeled axes). |
| `map(x, "fn")` | 2 | Apply a math built-in (by string name, e.g. `"sigmoid"`, `"exp"`) element-wise while preserving labels. |

Annotation syntax on assignment attaches labels in one step:

```
X : [batch, feat] = randn(7, [60, 2])
Q : [seq, d_k] = randn(17, [6, 4])
```

Labels propagate through elementwise ops (one-None / one-Some
accepted), matmul (contraction axis validated, outer dims passed
through), reductions (the reduced axis's label drops), and `map()`.
A mismatch surfaces as a structured
`EvalError::ShapeMismatch { op, expected, actual }` whose Display
renders both labeled shapes side by side.

### Autograd

| Function | Args | Description |
|----------|------|-------------|
| `grad(expr, wrt)` | 2 | Lift `expr` onto the reverse-mode tape and return the gradient wrt the named parameter or tensor. Shape equals the shape of `wrt`. Supported ops: `+`, `-`, `*`, `/`, unary `-`, `exp`, `log`, `sigmoid`, `tanh_fn`, `relu` (via `relu_layer`), `softmax`, `sum` / `reduce_add`, `mean`, `transpose`, `reshape`, `matmul`, `cross_entropy`. Use with `param[shape]` / `tensor[shape]` leaves. |

### Optimizers and Schedules

| Function | Args | Description |
|----------|------|-------------|
| `momentum_sgd(loss, params, lr, beta)` | 4 | One in-place momentum-SGD step on `params`. `params` is a single param name, a `[p1, p2, ...]` list, or a model identifier (walked via `params(model)`). Per-parameter state is maintained on the environment so the next call continues the trajectory. |
| `adam(loss, params, lr, b1, b2, eps)` | 6 | One Adam step on `params`. Same `params` shape as `momentum_sgd`; per-parameter `m`/`v` state is maintained across calls. |
| `cosine_schedule(step, total, lr_min, lr_max)` | 4 | Cosine annealing from `lr_max` at `step=0` to `lr_min` at `step=total`. Pure scalar helper usable inside `adam(..., cosine_schedule(step, 100, 1e-4, 1e-2), ...)`. |
| `linear_warmup(step, warmup, lr)` | 3 | Ramp from 0 to `lr` over the first `warmup` steps and return `lr` after. |
| `params(model)` | 1 | Return the flat list of parameter names owned by a model; used internally by the optimizers when given a model identifier. |

### Model DSL

Models are a `Value::Model` runtime value built by composition. A
"parameterless" layer still carries state (the owned parameters it
initialized at construction). Apply a model to an array with
`apply(model, X)`; gradients flow back through every owned parameter.

| Function | Args | Description |
|----------|------|-------------|
| `linear(in, out, seed)` | 3 | Seeded `W : [in, out]` + `b : [out]`, `apply(m, X)` computes `X W + b`. |
| `tanh_layer()` | 0 | Parameter-free `tanh_fn`. |
| `relu_layer()` | 0 | Parameter-free `relu` activation (zeros negatives). |
| `softmax_layer()` | 0 | Parameter-free `softmax(x, last_axis)`. |
| `rms_norm(dim)` | 1 | Per-row RMS normalization: `y[i] = x[i] / sqrt(mean(x[i]^2) + 1e-8)`. |
| `chain(a, b, ...)` | Nx | Sequential composition: `apply(chain(a, b, c), X) = apply(c, apply(b, apply(a, X)))`. |
| `residual(block)` | 1 | Skip connection: `apply(residual(b), X) = apply(b, X) + X`. The inner block must preserve input shape. |
| `attention(d_model, heads, seed)` | 3 | Multi-head self-attention. Input `[T, d_model]` (or `[B, T, d_model]`), output same shape. Tape-lowered for `heads=1`, forward-only for `heads>1`. |
| `causal_attention(d_model, heads, seed)` | 3 | Same as `attention` but applies a lower-triangular mask (upper-triangle scores become `-1e9` before softmax) so position `t` cannot attend to `t+k` for `k > 0`. Tape-lowered for `heads=1`. |
| `embed(vocab_size, d_model, seed)` | 3 | Learned `[vocab, d_model]` lookup table. `apply(embed, tokens)` where `tokens` is a rank-1 `[T]` (or rank-2 `[B, T]`) integer array returns `[T, d_model]` (or `[B, T, d_model]`). Gradients accumulate on the embedding rows touched by `tokens`. |
| `sinusoidal_encoding(seq_len, d_model)` | 2 | Deterministic `[time=seq_len, dim=d_model]` sinusoidal positional table. No parameters. Additive pattern: `apply(embed, toks) + sinusoidal_encoding(T, d)`. |
| `apply(model, X)` | 2 | Forward pass. For `embed`, `X` is integer tokens; for everything else it is an `[..., d_in]` float array. Fully differentiable through the tape. |

### Data Loading and Dataset Prep

| Function | Args | Description |
|----------|------|-------------|
| `load(path)` | 1 | Terminal REPL only (`--data-dir <path>` required). `"foo.csv"` returns a labeled `DenseArray` of the CSV's numeric columns; `"foo.txt"` (or any non-CSV extension) returns a whole-file `Value::Str`. Absolute and traversing paths are rejected. |
| `load_preloaded(name)` | 1 | Returns a compiled-in corpus as a `Value::Str`. Current registry: `"tiny_corpus"` (short pangram-style text) and `"tiny_shakespeare_snippet"` (~KB of Shakespeare). Works in both REPLs. |
| `shuffle(x, seed)` | 2 | Fisher-Yates row permutation on a rank>=1 array. Labels preserved. Deterministic for a given seed. |
| `batch(x, size)` | 2 | Return a rank-(r+1) array of contiguous row batches; the last batch is zero-padded if `n_rows` is not divisible by `size`. |
| `batch_mask(x, size)` | 2 | Return the 0/1 mask matching `batch(x, size)` (1 for real rows, 0 for padded). |
| `split(x, train_frac, seed)` | 3 | Return the first `round(train_frac * n_rows)` rows after a deterministic shuffle. |
| `val_split(x, train_frac, seed)` | 3 | Companion to `split`; returns the complementary rows with the same seed. |

### Tokenizers

| Function | Args | Description |
|----------|------|-------------|
| `tokenize_bytes(s)` | 1 | Return a rank-1 array of byte indices (0-255) for the UTF-8 encoding of `s`. Pure, deterministic, no training. |
| `decode_bytes(tokens)` | 1 | Inverse of `tokenize_bytes`; returns a `Value::Str`. |
| `train_bpe(corpus, vocab_size, seed)` | 3 | Train a byte-level BPE tokenizer on a `Value::Str` (or already-byte-tokenized rank-1 array). Returns a `Value::Tokenizer`. Deterministic tie-breaking: on ties in merge count, the lexicographically smallest byte pair wins. |
| `apply_tokenizer(tok, text)` | 2 | Encode `text` (a `Value::Str`) through a trained tokenizer; returns a rank-1 integer array. |
| `decode(tok, tokens)` | 2 | Inverse of `apply_tokenizer`. For every byte string `s`, `decode(tok, apply_tokenizer(tok, s)) == s`. |

### Language Model Helpers

| Function | Args | Description |
|----------|------|-------------|
| `shift_pairs_x(ids, block_size)` | 2 | Build next-token-prediction input windows from a 1-D token array. Returns an `[N, block_size]` integer matrix where each row is a contiguous window of `ids`. |
| `shift_pairs_y(ids, block_size)` | 2 | Matching target windows for `shift_pairs_x`: each row is the input window shifted right by one position. |
| `last_row(M)` | 1 | Return the last row of a rank-2 matrix as a rank-1 vector. Used in generation loops to extract the final position's logits from an `[T, V]` model output. |
| `concat(a, b)` | 2 | Concatenate two rank-0 or rank-1 arrays into a 1-D vector. Used in generation loops to append a sampled token id to the growing sequence. |
| `attention_weights(model, X)` | 2 | Read-only forward pass that walks `model` to its first `attention` / `causal_attention` layer, transforms `X` through any preceding layers in the outer chain, and returns the softmax attention weight matrix (`[T, T]` single-head or `[heads, T, T]` multi-head). Renders well as a heatmap. |

### Experiments

| Function | Args | Description |
|----------|------|-------------|
| `compare(name_a, name_b)` | 2 | Return a `Value::Str` with a side-by-side view of the most-recent runs with those names, including per-metric deltas. Merges memory-only (web REPL) and on-disk (terminal REPL, under `--exp-dir`) records. |

### Visualization

| Function | Args | Description |
|----------|------|-------------|
| `svg(data, type)` | 2 | Render `data` as an SVG diagram of the given type and return the SVG string |
| `svg(data, type, aux)` | 3 | Same, with an auxiliary array (used by `decision_boundary`) |

Supported `type` values:

- `"scatter"` -- expects an Nx2 matrix; one circle per row.
- `"line"` -- a vector becomes a polyline; an Nx2 matrix becomes (x,y) points connected by lines.
- `"bar"` -- a vector becomes a bar chart with one bar per element.
- `"heatmap"` -- an MxN matrix rendered as a viridis-colored grid.
- `"decision_boundary"` -- a 2D classifier-output grid rendered as a diverging-color surface, with the third argument as an Nx3 `[x, y, label]` training matrix overlaid as colored points.

The browser REPL detects SVG return values and renders them inline.
The CLI REPL prints a `[svg: N bytes]` summary; pass `--svg-out <dir>`
to write each SVG to a file.

### Analysis helpers

High-level helpers that compute and render a complete diagram in
one call. Each returns an SVG string just like `svg()`.

| Function | Args | Description |
|----------|------|-------------|
| `hist(data, bins)` | 2 | Histogram of a vector with `bins` equal-width bins, rendered as a bar chart |
| `scatter_labeled(points, labels)` | 2 | Nx2 points colored by a length-N cluster-id vector |
| `loss_curve(losses)` | 1 | Vector of losses rendered as a line plot with axis labels |
| `confusion_matrix(predicted, actual)` | 2 | KxK heatmap of class-id predictions vs actual labels with cell counts overlaid |
| `boundary_2d(grid_outputs, dims, points, labels)` | 4 | Render a 2D classifier surface from a length-(rows*cols) vector and `[rows, cols]` dims, with separately-supplied training points and labels |

## Array Display

Arrays are displayed in a row-major layout:

```
mlpl> 42
42
mlpl> [1, 2, 3]
1 2 3
mlpl> reshape(iota(6), [2, 3])
0 1 2
3 4 5
```

Scalars print as a single number. Vectors print space-separated on
one line. Matrices print one row per line.

## Broadcasting Rules

When an operator combines a scalar with an array, the scalar is
broadcast to match the array's shape:

```
[1, 2, 3] + 10     # [11, 12, 13]
5 * [1, 2, 3]       # [5, 10, 15]
```

When both operands are arrays, they must have the same shape.

## Error Handling

Errors are reported inline with descriptive messages:

- Shape mismatches: "shapes do not match: [2, 3] vs [3, 2]"
- Unknown functions: "unknown function: foo"
- Arity errors: "reshape expects 2 arguments, got 1"
- Undefined variables: "undefined variable: x"
