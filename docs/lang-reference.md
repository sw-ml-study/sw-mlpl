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

## Builtin / Operator References (`:foo`)

```
:add
:max
:+
:*
:sigmoid
```

A first-class-ish reference to a builtin or operator. Lexed as
`:` immediately followed (no intervening space) by an
identifier-start character or one of `+ * / -`. Evaluates to a
`Value::BuiltinRef { name }`. Used as the first argument to
higher-order builtins like `reduce(:op, x[, axis])`. Lives in a
separate namespace from regular variables, so `add = 42` cannot
shadow `:add`. Variables can hold a BuiltinRef:

```
f = :max
reduce(f, [3, 1, 4, 1, 5, 9, 2, 6])    # 9
```

The annotation form (`x : [batch] = ...`) requires a space after
`:` and is unaffected. The design is forward-compatible with
first-class functions: `:foo` can lift to a function value
cleanly.

## The Three Kinds of Name: `name` vs `:name` vs `u:name`

One naming design, three deliberate roles:

| Form | What it is | Example |
| --- | --- | --- |
| `name(...)` | CALL a builtin. The bare name is only meaningful applied to arguments. | `disp(G)`, `rotate(x, 1, 0)` |
| `:name` | QUOTE a builtin into a first-class VALUE (APL's quoted function). What higher-order builtins consume; typing `:disp` alone shows the reference, not a rendering. Applying the quoted form calls it: `:disp(G)` and `disp(G)` are the same call. `:disp G` (no parens) is NOT a command -- the REPL says so. | `reduce(:add, S, 0)`, `f = :max`, `:disp(G)` |
| `u:name` | A USER-DEFINED function. The mandatory `u:` namespace prefix keeps your names from ever colliding with (or shadowing) present or FUTURE builtins -- a new builtin release can never break your workspace. | `def u:life(g) { ... }`, `u:life(G)` |

Why the trichotomy: calls and values must look different (so
`reduce(add, v)` cannot silently pass a VARIABLE named add where
a function was meant), and user space and builtin space must
look different (APL workspaces suffered decades of name-clash
pain; the `u:` prefix is the cure priced at two characters).
The quoting rule covers BOTH spaces: `:name` quotes a builtin
and `:u:name` quotes YOUR function -- a first-class reference
you can bind, store in record fields, pass, and return. A test
registry is a record of references. `call(f, args...)` invokes
either kind uniformly, with errors naming the REFERENCED
function; the Result combinators `map_ok` / `and_then` /
`or_else` take a reference as their first argument. References
identify definitions by NAME (late binding): re-defining
`u:name` means existing references call the new definition.

Introspection follows the same split: `:builtins` lists the
builtin space, `:fns` lists YOUR `u:` space, `:list u:name`
prints a definition back (verbatim source, `#` comments
included, when defined in this process), and `:describe` works
on both.

## Value Kinds

Every value the evaluator produces is one of NINE kinds (the
`kind` field connect-mode responses carry uses the same names):

| Kind | Produced by | Notes |
| --- | --- | --- |
| `array` | literals, range/fill/randn, all math | The workhorse: rank-N dense f64 tensor, optional axis labels. |
| `builtin-ref` | `:name` | The quoted-builtin value above. |
| `device-tensor` | `device("mlx"/"cuda") { ... }` results held on-device | Fetched back on demand; keeps GPU residency explicit. |
| `model` | linear/chain/attention/... model DSL | Layer tree; renders as a Sankey diagram. |
| `record` | `{field: expr, ...}` literals | Named fields, `r.field` access; the canonical error-object payload. |
| `result` | `ok(v)` / `err(e)`, to_number/to_int/env, list_get | The two-state sum consumed by is_ok/unwrap/unwrap_or/get_value/get_error/`?` (docs/error-handling.md). |
| `string` | `"..."` literals, svg()/disp() and other renderers | Separate from arrays; `x = "hi"` binds a string. |
| `string-list` | `["a", "b"]` (all-string literals), tokenizer vocab helpers | Sibling of array for text data. |
| `tokenizer` | train_bpe | Paired with apply_tokenizer/decode. |

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
range(6)
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
for row in reshape(range(6), [3, 2]) { reduce_add(row) }
last_rows   # [1, 5, 9]
```

## Include Declaration

```text
include "helpers.mlpl"
```

Top-level, script-mode static source loading: the file's
statements splice in at the include site, in source order, with
duplicate loads ignored and cycles reported with the full
chain. The argument must be a string literal; `include` stays a
legal variable name everywhere else (only the
literal-string-at-top-level form is an include). Resolution is
sandboxed under `--source-dir` (default: the script's
directory); see the Usage Guide's "Including Source Files"
section for the rules. Surfaces without a source provider (the
browser session, connect mode, the interactive prompt) reject
the declaration with a precise error.

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

### Streaming metrics (connect mode)

When a program runs on a connected `mlpl-serve` peer, every `train`
iteration also streams metrics to the client as SSE `metric` frames,
driving the web playground's live loss panel and LOSS sparkline row:

- every scalar bound to a `*_metric` name inside the block streams
  per step under that name (a `val`-prefixed name, e.g.
  `val_loss_metric`, charts as the validation series);
- when the block binds NO `*_metric` name, the block's own per-step
  loss (its final expression, the same scalar that fills
  `last_losses`) streams as the implicit `loss` metric -- so a plain
  `train N { ... }` shows a live loss curve with no extra code and no
  recompute.

## Labeled Axes

An assignment may carry axis labels as metadata so label mismatches
in downstream ops surface with a `ShapeMismatch` error that names
both labeled shapes. Labels propagate through elementwise ops,
matmul (contraction axis validated), reductions (the reduced axis's
label drops), and `map()`.

```
M : [batch, feat] = reshape(range(6), [2, 3])
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

## Table Ordering Policy

Lookup tables and listings in this reference (and in `:builtins` /
`:vars` / `:fns` output) sort ALPHABETICALLY within their group --
groups exist only where they aid discovery. Time-ordered logs
(`:experiments`, CHANGES) stay chronological; short narrative
tables (e.g. the three name forms) keep their teaching order.

## Built-in Functions

### Array Operations

| Function | Args | Description |
|----------|------|-------------|
| `argtop_k(scores, k)` | 2 | Indices of the top-`k` entries of a rank-1 `scores` vector, sorted by descending score (ties go to the lower index). Used to pick the strongest variants in ensemble / Neural-Thicket workflows. |
| `running_product(v)` | 1 | Running product along a rank-1 vector (e.g. a diffusion noise schedule's alpha-bar). `cumprod` is the deprecated alias. |
| `grade_up(v)` | 1 | Stable argsort, ascending: the index vector that sorts rank-1 `v` (ties keep original order). `gather_rows(X, grade_up(d))` reorders a dataset by difficulty -- the curriculum idiom. |
| `grade_down(v)` | 1 | Stable argsort, descending. `gather_rows(C, grade_down(scores))` ranks candidates best-first; `take(grade_down(s), 0, 0)` is the best index. |
| `call(f, args...)` | 1+ | Invoke a function REFERENCE uniformly: `call(:u:double, 21)` runs your function, `call(:mean, v)` runs the builtin. Errors identify the referent (arity, unknown name), and `ok`/`err`/`?` semantics flow through unchanged. |
| `map_ok(f, r)` | 2 | Apply `f` inside `ok(...)`: `ok(x)` becomes `ok(f(x))`; `err` passes through untouched. `f` is a function reference; builtin references need an array payload. |
| `and_then(f, r)` | 2 | Chain fallible steps (the railway): `ok(x)` becomes `f(x)` where `f` itself returns a Result; `err` passes through. |
| `or_else(f, r)` | 2 | Recover: `err(e)` becomes `f(e)`; `ok` passes through untouched. |
| `equal(a, b)` | 2 | Total STRUCTURAL equality over any two values: numbers (IEEE, except NaN equals NaN), strings, arrays (shape + axis labels + elements), records (recursing), models, tokenizers, `ok`/`err` results. Mismatched kinds return 0 -- never a hard error -- so assertions stay honest. Returns scalar 1/0. |
| `repr(v)` | 1 | Deterministic, BOUNDED rendering of any value for expected-vs-actual diagnostics: `array[2, 3] [0, 1, ...]` with labels, quoted escaped strings, `{field: ...}` records, `ok(...)`/`err(...)`. Truncates large values with an explicit marker; not a serialization format. |
| `compress(mask, a[, axis])` | 2-3 | Keep the slices of `a` along `axis` (default 0) where rank-1 `mask` is nonzero (APL compress). `compress(gt(scores, t), C)` keeps verified candidates; works on any rank. |
| `pareto_plot(P, dirs)` | 2 | Render the frontier: every metric pair as a dot (frontier members highlighted), plus the staircase line through the frontier. Computes the mask with `pareto_front` internally, so plot and mask always agree. |
| `pareto_front(P, dirs)` | 2 | The `[n]` 0/1 mask of non-dominated rows of the `[n, k]` metric matrix `P`; `dirs` is `[k]` with `1` = maximize the column, `-1` = minimize. `compress(pareto_front(P, dirs), P)` keeps the frontier; `scatter_labeled(P, pareto_front(P, dirs))` highlights it. |
| `kg_neighbors(edges, node[, rel])` | 2-3 | One-hop destination ids from `node` in an `[E, 3]` `(src, rel, dst)` edge array, sorted and deduplicated; the optional third argument restricts to one relation. |
| `kg_verify(edges, paths)` | 2 | Row-batched path checker: for `[n, L]` id sequences, `out[i]` is 1 when every consecutive pair in row i is an edge (any relation). Rank-1 input is one path. The answer oracle for multi-hop tasks. |
| `kg_paths(edges, hops, n, seed)` | 4 | `[n, hops+1]` valid paths sampled by seeded random walk (uniform start edge, uniform outgoing edge; dead ends restart). Every sampled row passes `kg_verify`. The multi-hop task generator. |
| `kg_split(edges, frac, seed)` | 3 | Entity-disjoint `{seen, unseen}` record: entities are shuffled by seed, the first `frac` become the seen set; an edge is `seen` only if BOTH endpoints are seen entities, so unseen paths must visit entities the seen side never contains. |
| `rand_ints(n, lo, hi, seed)` | 4 | `[n]` uniform integers in `[lo, hi)`, deterministic per seed (explicit PRNG state, same bits everywhere). The integer source for synthetic-data generators; pairs with `randn` (Gaussian floats). |
| `dedupe_rows(X)` | 1 | Unique rows of a rank-2 `[n, L]` array (first occurrence kept, original order) as a `{rows, index}` record: `d.rows` for direct use, `gather_rows(Y, d.index)` to carry companion arrays along. |
| `running_sum(v)` | 1 | Running sum along a rank-1 vector: `out[i]` is the sum of `v[0..=i]` (prefix sums, CDFs, cumulative totals). The additive sibling of `running_product`. Rank-1 only: focus a row/column of a higher-rank value with `take(a, axis, i)`, or scan the whole array explicitly with `flatten(a)`. |
| `depth(a)` | 1 | Nesting depth (scalar): `0` for a scalar, `1` for any array. APL heritage. |
| `disp(a)` | 1 | Returns an ASCII box diagram (a `Value::Str`) that makes the rank, shape, and depth of `a` visible: rank <= 2 as a framed grid, rank >= 3 as a labeled stack of leading-axis slices, plus a `rank R  shape [..]  depth D` footer. MLPL's answer to APL's `]display`. |
| `emit_frame(name, step, x)` | 3 | Stream tensor `x` as a live frame through the connect-mode metric sink (the whole-tensor analog of `_metric` scalars); a no-op when not connected. Returns `x`. |
| `get_error(r)` | 1 | The Err side of a Result as a 0-or-1 element vector: `[e]` when Err, `[]` when Ok. Complementary to `get_value` by construction. |
| `get_value(r)` | 1 | The Ok side of a Result as a 0-or-1 element vector: `[v]` when `r` is `Ok(v)` (scalar payloads), `[]` when Err. `tally` of it is `is_some`; APL2 zilde-Option flavor (see docs/error-handling.md). |
| `linspace(start, stop, n)` | 3 | `n` evenly-spaced values from `start` to `stop` inclusive, as a rank-1 vector. |
| `range(n)` | 1 | Integers 0, 1, ..., n-1 as a vector (preferred name) `iota(n)` is a DEPRECATED alias (APL heritage): it still evaluates but is absent from docs and examples; prefer `range`. |
| `rank(a)` | 1 | Number of dimensions (scalar) |
| `reduce(:op, a)` | 2 | Higher-order reduction: `:op` is one of `:add`/`:+`, `:mul`/`:*`, `:min`, `:max`, `:and`, `:or`. Examples: `reduce(:max, v)`, `reduce(:and, mask)`. The first argument is a `BuiltinRef` (`:foo` syntax); user variables can hold one too: `f = :max; reduce(f, v)`. |
| `ngram_hash(ids, orders, heads, slots, seed)` | 5 | Rolling n-gram hash indices `[T, order, head]` for Engram-style memory addressing; a frozen exact cross-backend contract (ids capped at 2^21 - 1). |
| `reduce(:op, a, axis)` | 3 | Same, restricted to a single axis. |
| `reduce_add(a[, axis])` | 1-2 | Sum all elements (or along axis). Equivalent to `reduce(:add, a[, axis])`; kept as a direct shorthand. |
| `reduce_mul(a[, axis])` | 1-2 | Product. Equivalent to `reduce(:mul, a[, axis])`. |
| `apply_engram(e, h, ids)` | 3 | Engram forward pass: hash the ids, gather the addressed memory rows, project, concat-gate against `h`, and add to the residual stream. Exact no-op on a freshly built engram; differentiable, so `grad`/`adam`/`train` move only the addressed memory rows (duplicates accumulate). |
| `engram(hidden, ngrams, heads, slots, head_dim, seed)` | 6 | Engram conditional-memory layer: one flattened n-gram table + value projection + concat gate, initialized near-identity (zero table, gate bias -2). Apply with `apply_engram`; trainable via `adam(loss, e, ...)`. |
| `engram_stats(e, ids)` / `engram_stats(e, h, ids)` | 2-3 | Engram health record with addressable fields: `rows_addressed` (total (t, order, head) lookups), `unique_rows`, `collisions` (distinct n-gram contexts sharing a slot under the frozen hash contract -- repetition of the same context is not a collision), `nonzero_rows`, `max_row_norm`; the 3-argument form adds `gate_mean` / `gate_max` from the eager forward's gate. Example: `s = engram_stats(e, ids); s.collisions`. |
| `gather_rows(table, indices)` | 2 | Select whole rows of a rank-2 table; output shape is the indices' shape + `[dim]`. Out-of-range indices error loudly. |
| `reshape(a, dims)` | 2 | Reshape array to new dimensions |
| `flatten(a)` | 1 | Ravel: all elements as a rank-1 vector in row-major order (equivalent to `reshape(a, [size(a)])`). Naming policy: meaningful names are canonical and arity-locked; APL glyph names are heritage aliases only. |
| `rotate(x, k, axis)` | 3 | Cyclic shift along axis; negative k (spell it `0 - k`) rotates the other way |
| `scatter(buffer, index, value)` | 3 | A copy of rank-1 `buffer` with the single entry at `index` replaced by `value` (the input is not mutated). The bulk form is a `u:stamp`-style loop (see the Life demos). |
| `shape(a)` | 1 | Dimension vector of array |
| `size(a)` | 1 | Total element count (scalar): the product of the shape (numel). A scalar has size `1`; `size(reshape(range(6), [2, 3]))` is `6`. |
| `tally(a)` | 1 | Length of the leading axis (scalar): the number of major cells (APL's monadic tally, J's `#`). A scalar tallies to `1`; a rank >= 1 array tallies to `shape[0]`. Contrast with `size`, which counts every element. |
| `transpose(a)` | 1 | Reverse axis order |

#### Tensor terminology bridge

The structural-introspection builtins come from the APL/APL2 lineage,
but every one maps onto a familiar tensor concept. The names mostly
agree across ecosystems -- with two traps worth memorizing (marked
below).

| MLPL builtin | Tensor concept | NumPy | PyTorch | Note |
|--------------|----------------|-------|---------|------|
| `depth(a)` | nesting level | -- | -- | TRAP: not the same as rank. A rank-5 *dense* tensor still has depth 1; depth only exceeds 1 for ragged / nested arrays (cf. `RaggedTensor`, `NestedTensor`), which arrive in a later stage. |
| `disp(a)` | structural pretty-print | `repr(a)` | `print(a)` | APL's `]display`. Note MATLAB's `disp` just prints values; MLPL's draws a box diagram framing rank / shape / depth. |
| `rank(a)` | number of axes | `.ndim` | `.dim()` | Not linear-algebra rank (column-span dimension) or LoRA rank (adapter inner dim). |
| `shape(a)` | shape / dims | `.shape` | `.shape` | Clean match everywhere. |
| `size(a)` | element count (numel) | `.size` | `.numel()` | TRAP: PyTorch `.size()` and MATLAB `size()` return the *shape*, not the count. MLPL follows NumPy / TensorFlow: `size` is numel. |
| `tally(a)` | leading-axis length | `len(a)` | `.size(0)` | The count of major cells: usually the batch size `N`, sequence length, or row count. |

Rules of thumb: `rank` counts axes, `shape` sizes them, `size` counts
every element, `tally` counts rows (the leading axis), and `depth`
counts levels of nesting (not axes).

### Linear Algebra

| Function | Args | Description |
|----------|------|-------------|
| `dot(a, b)` | 2 | Dot product of two vectors |
| `matmul(a, b)` | 2 | Matrix multiplication |

### Math Functions

| Function | Args | Description |
|----------|------|-------------|
| `abs(a)` | 1 | Element-wise absolute value |
| `ceil(a)` | 1 | Element-wise ceiling (round toward positive infinity) |
| `cos(a)` | 1 | Element-wise cosine (radians) |
| `e()` | 0 | The constant 2.71828182845904... |
| `exp(a)` | 1 | Element-wise e^x |
| `floor(a)` | 1 | Element-wise floor (round toward negative infinity) |
| `log(a)` | 1 | Element-wise natural logarithm |
| `mod(a, b)` | 2 | Element-wise remainder (a % b). Broadcasts like other binary ops. |
| `pi()` | 0 | The constant 3.14159265358979... |
| `pow(a, b)` | 2 | Element-wise a^b |
| `relu(a)` | 1 | Element-wise max(0, a). Standalone version of the model-DSL relu_layer(). |
| `round(a)` | 1 | Element-wise round to nearest integer |
| `sin(a)` | 1 | Element-wise sine (radians) |
| `sqrt(a)` | 1 | Element-wise square root |

### ML Activations

| Function | Args | Description |
|----------|------|-------------|
| `sigmoid(a)` | 1 | Logistic sigmoid: 1 / (1 + exp(-x)) |
| `tanh_fn(a)` | 1 | Hyperbolic tangent |

### Comparison and Statistics

| Function | Args | Description |
|----------|------|-------------|
| `eq(a, b)` | 2 | Element-wise equality (returns 0 or 1) |
| `gt(a, b)` | 2 | Element-wise greater-than (returns 0 or 1) |
| `lt(a, b)` | 2 | Element-wise less-than (returns 0 or 1) |
| `mean(a)` | 1 | Arithmetic mean of all elements |

### Array Constructors

| Function | Args | Description |
|----------|------|-------------|
| `argmax(a)` | 1 | Scalar index of the maximum element over all elements of `a` (flat). |
| `argmax(a, axis)` | 2 | Index (as f64) of the max along `axis`; output rank is one less than input. Ties go to the first occurrence. |
| `blobs(seed, n_per_class, centers)` | 3 | Seeded 2D gaussian-blob dataset. `centers` is a Kx2 matrix (or length-2K vector) of cluster centers; returns an Nx3 matrix where each row is `[x, y, label]`, with `N = K * n_per_class` and noise sigma 0.15. |
| `circles(seed, n_per_class, noise)` | 3 | Seeded two-concentric-circles dataset, same `Nx3` layout as `moons`. |
| `cross_entropy(logits, targets)` | 2 | Scalar mean negative log-likelihood. `logits` is `[N, V]` or `[B, T, V]`; `targets` is `[N]` or `[B, T]` integer-valued. Fused, numerically-stable log-softmax + NLL; fully differentiable wrt `logits` via `grad(...)`. |
| `fill(shape, v)` | 2 | Array filled with value v |
| `grid(bounds, n)` | 2 | n*n by 2 matrix of (x, y) points over [xmin,xmax] x [ymin,ymax]; bounds is [xmin, xmax, ymin, ymax] |
| `moons(seed, n_per_class, noise)` | 3 | Seeded two-moons dataset; returns an `Nx3` matrix of `[x, y, label]` for `N = 2 * n_per_class`. |
| `one_hot(labels, k)` | 2 | Convert a length-N label vector to an `NxK` one-hot matrix. |
| `ones(shape)` | 1 | Array of ones with given shape |
| `perplexity(logits, targets)` | 2 | Convenience: `exp(cross_entropy(logits, targets))`. The canonical language-model evaluation metric (lower is better). Same arg shapes as `cross_entropy`. Forward-only -- use `cross_entropy` directly inside `grad(...)`. |
| `randn(seed, shape)` | 2 | Seeded standard-normal array (mean 0, variance 1), same shape semantics as `random`. Implemented via Box-Muller on the same xorshift64 stream. |
| `random(seed, shape)` | 2 | Seeded uniform [0, 1) array with the given shape. `seed` is a scalar integer; `shape` is a vector of dimensions. Deterministic for a given seed. |
| `sample(logits, temperature, seed)` | 3 | Categorical sample from a 1-D `[V]` logit vector. Returns a scalar integer token id. `temperature == 0.0` collapses to `argmax(logits)`; otherwise draws from `softmax(logits / temperature)` via inverse-CDF on a single seeded uniform. Same `(logits, temperature, seed)` always yields the same id. |
| `softmax(a, axis)` | 2 | Softmax along `axis`, stabilized by subtracting the per-group max before exponentiation. |
| `top_k(logits, k)` | 2 | Return a `[V]` logit vector with all but the top-`k` entries replaced by `-inf`. Pure (no randomness). Compose with `sample` for top-k sampling: `sample(top_k(logits, k), temperature, seed)`. |
| `zeros(shape)` | 1 | Array of zeros with given shape |

### Labeled Axes

| Function | Args | Description |
|----------|------|-------------|
| `label(x, names)` | 2 | Attach axis labels to an array. `names` is a rank-1 string array; length must equal the rank of `x`. Use `""` for "no label" on a single axis. |
| `labels(x)` | 1 | Return the axis labels of `x` as a comma-joined string ("" for unlabeled axes). |
| `map(x, "fn")` | 2 | Apply a math built-in (by string name, e.g. `"sigmoid"`, `"exp"`) element-wise while preserving labels. |
| `relabel(x, names)` | 2 | Like `label`, but explicitly overrides any existing labels on `x`. |
| `reshape_labeled(x, dims, names)` | 3 | Combine `reshape` and `label` in one call. New axes get the given names; plain `reshape` clears labels. |

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
| `adam(loss, params, lr, b1, b2, eps)` | 6 | One Adam step on `params`. Same `params` shape as `momentum_sgd`; per-parameter `m`/`v` state is maintained across calls. |
| `cosine_schedule(step, total, lr_min, lr_max)` | 4 | Cosine annealing from `lr_max` at `step=0` to `lr_min` at `step=total`. Pure scalar helper usable inside `adam(..., cosine_schedule(step, 100, 1e-4, 1e-2), ...)`. |
| `linear_warmup(step, warmup, lr)` | 3 | Ramp from 0 to `lr` over the first `warmup` steps and return `lr` after. |
| `momentum_sgd(loss, params, lr, beta)` | 4 | One in-place momentum-SGD step on `params`. `params` is a single param name, a `[p1, p2, ...]` list, or a model identifier (walked via `params(model)`). Per-parameter state is maintained on the environment so the next call continues the trajectory. |
| `params(model)` | 1 | Return the flat list of parameter names owned by a model; used internally by the optimizers when given a model identifier. |

### Model DSL

Models are a `Value::Model` runtime value built by composition. A
"parameterless" layer still carries state (the owned parameters it
initialized at construction). Apply a model to an array with
`apply(model, X)`; gradients flow back through every owned parameter.

| Function | Args | Description |
|----------|------|-------------|
| `apply(model, X)` | 2 | Forward pass. For `embed`, `X` is integer tokens; for everything else it is an `[..., d_in]` float array. Fully differentiable through the tape. |
| `attention(d_model, heads, seed)` | 3 | Multi-head self-attention. Input `[T, d_model]` (or `[B, T, d_model]`), output same shape. Tape-lowered for `heads=1`, forward-only for `heads>1`. |
| `causal_attention(d_model, heads, seed)` | 3 | Same as `attention` but applies a lower-triangular mask (upper-triangle scores become `-1e9` before softmax) so position `t` cannot attend to `t+k` for `k > 0`. Tape-lowered for `heads=1`. |
| `chain(a, b, ...)` | Nx | Sequential composition: `apply(chain(a, b, c), X) = apply(c, apply(b, apply(a, X)))`. |
| `embed(vocab_size, d_model, seed)` | 3 | Learned `[vocab, d_model]` lookup table. `apply(embed, tokens)` where `tokens` is a rank-1 `[T]` (or rank-2 `[B, T]`) integer array returns `[T, d_model]` (or `[B, T, d_model]`). Gradients accumulate on the embedding rows touched by `tokens`. |
| `linear(in, out, seed)` | 3 | Seeded `W : [in, out]` + `b : [out]`, `apply(m, X)` computes `X W + b`. |
| `predict_batch(model, X)` | 2 | Forward pass + `argmax` over the trailing axis. Returns integer class indices. Not differentiable -- use `apply(model, X)` inside `grad()` or `adam()` instead. Convenient for evaluation: `preds = predict_batch(mdl, X); accuracy = reduce_add(eq(preds, Y)) / N`. |
| `relu_layer()` | 0 | Parameter-free `relu` activation (zeros negatives). |
| `residual(block)` | 1 | Skip connection: `apply(residual(b), X) = apply(b, X) + X`. The inner block must preserve input shape. |
| `rms_norm(dim)` | 1 | Per-row RMS normalization: `y[i] = x[i] / sqrt(mean(x[i]^2) + 1e-8)`. |
| `sinusoidal_encoding(seq_len, d_model)` | 2 | Deterministic `[time=seq_len, dim=d_model]` sinusoidal positional table. No parameters. Additive pattern: `apply(embed, toks) + sinusoidal_encoding(T, d)`. |
| `softmax_layer()` | 0 | Parameter-free `softmax(x, last_axis)`. |
| `tanh_layer()` | 0 | Parameter-free `tanh_fn`. |

### CNN + RNN builtins

| Function | Args | Description |
|----------|------|-------------|
| `conv2d(input, filters, stride, padding)` | 4 | 2D convolution. `input`: `[B,C_in,H,W]`, `filters`: `[C_out,C_in,kH,kW]`. `stride` and `padding` are scalars. Returns `[B,C_out,H_out,W_out]`. |
| `lstm_cell(input, hidden, cell, W, bias)` | 5 | One LSTM step. `W`: `[4*hd, id+hd]`, `bias`: `[4*hd, 1]`. Returns `[2*hd, 1]` = concat(new_hidden, new_cell). Split with `reshape` + `take`. |
| `pool2d(input, size, mode)` | 3 | 2D pooling. `mode=1` max pooling, `mode=0` average pooling. `size` is the square pool window side. |
| `rnn_cell(input, hidden, W_ih, W_hh, bias)` | 5 | One Elman RNN step: `tanh(W_ih @ input + W_hh @ hidden + bias)`. Returns updated hidden state. |

### Records

`{name: expr, ...}` builds a record; `r.name` reads a field.
Several builtins return records so results stay addressable
(`engram_stats`, `dedupe_rows`, `kg_split`). Field names may be
any identifier INCLUDING language keywords -- `s.train`,
`{eval: 1}`, and `r.if` are legal, because the two member-name
positions (a record-literal key and the name after `.`) are
grammatically unambiguous. Keywords stay reserved everywhere
else. Duplicate keys are a parse error.

### Result type

A `Value::Result { ok, payload }` wraps success-or-failure for
ops that can fail without crashing the REPL. The payload can be
any Value (typically `Value::Array` for success and `Value::Str`
for error messages). Display is `Ok(<inner>)` or `Err(<inner>)`.

| Function | Args | Description |
|----------|------|-------------|
| `err(v)` | 1 | Wrap any value as `Err(v)`. Typically `err("message string")`. |
| `err_message(r)` | 1 | Return the payload if `Err(_)`. Raises `Unsupported` on `Ok(_)` (no message to return). |
| `is_err(r)` | 1 | Inverse of `is_ok`. |
| `is_ok(r)` | 1 | Return scalar `1.0` if `r` is `Ok(_)`, else `0.0`. Raises `NotAResult` on a non-Result first argument. |
| `ok(v)` | 1 | Wrap any value as `Ok(v)`. |
| `unwrap(r)` | 1 | Return the payload if `Ok(_)`. Raises `EvalError::UnwrapOnErr { message }` carrying the payload's display form if `Err(_)`. |
| `unwrap_or(r, default)` | 2 | Return the payload if `Ok(_)`; otherwise evaluate `default` and return that. |

First in-tree consumer: the `:upload x` web-REPL command binds `x = Ok({pixels: [1, 3, 64, 64], h: 64,
w: 64})` on a successful file pick and `x = Err("cancelled")`
when the user dismisses the dialog. After upload, branch on
`is_ok(x)` to classify the photo, or use `unwrap_or` to
substitute a default tensor. The classify pattern:

```mlpl
:upload x
img  = unwrap(x).pixels
pred = predict_batch(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(img, 16), [16, 768])), [1, 16, 128])), 1, 0))
```

Returns `[0]` for cat or `[1]` for dog after running one of
the trained Pets demos.

### Data Loading and Dataset Prep

| Function | Args | Description |
|----------|------|-------------|
| `batch(x, size)` | 2 | Return a rank-(r+1) array of contiguous row batches; the last batch is zero-padded if `n_rows` is not divisible by `size`. |
| `batch_mask(x, size)` | 2 | Return the 0/1 mask matching `batch(x, size)` (1 for real rows, 0 for padded). |
| `load(path)` | 1 | Terminal REPL only (`--data-dir <path>` required). `"foo.csv"` returns a labeled `DenseArray` of the CSV's numeric columns; `"foo.txt"` (or any non-CSV extension) returns a whole-file `Value::Str`. Absolute and traversing paths are rejected. |
| `load_preloaded(name)` | 1 | Returns a compiled-in corpus as a `Value::Str`. Current registry: `"tiny_corpus"` (short pangram-style text) and `"tiny_shakespeare_snippet"` (~KB of Shakespeare). Works in both REPLs. |
| `shuffle(x, seed)` | 2 | Fisher-Yates row permutation on a rank>=1 array. Labels preserved. Deterministic for a given seed. |
| `split(x, train_frac, seed)` | 3 | Return the first `round(train_frac * n_rows)` rows after a deterministic shuffle. |
| `val_split(x, train_frac, seed)` | 3 | Companion to `split`; returns the complementary rows with the same seed. |

### Tokenizers

| Function | Args | Description |
|----------|------|-------------|
| `apply_tokenizer(tok, text)` | 2 | Encode `text` (a `Value::Str`) through a trained tokenizer; returns a rank-1 integer array. |
| `decode(tok, tokens)` | 2 | Inverse of `apply_tokenizer`. For every byte string `s`, `decode(tok, apply_tokenizer(tok, s)) == s`. |
| `decode_bytes(tokens)` | 1 | Inverse of `tokenize_bytes`; returns a `Value::Str`. |
| `tokenize_bytes(s)` | 1 | Return a rank-1 array of byte indices (0-255) for the UTF-8 encoding of `s`. Pure, deterministic, no training. |
| `train_bpe(corpus, vocab_size, seed)` | 3 | Train a byte-level BPE tokenizer on a `Value::Str` (or already-byte-tokenized rank-1 array). Returns a `Value::Tokenizer`. Deterministic tie-breaking: on ties in merge count, the lexicographically smallest byte pair wins. |

### Language Model Helpers

| Function | Args | Description |
|----------|------|-------------|
| `attention_weights(model, X)` | 2 | Read-only forward pass that walks `model` to its first `attention` / `causal_attention` layer, transforms `X` through any preceding layers in the outer chain, and returns the softmax attention weight matrix (`[T, T]` single-head or `[heads, T, T]` multi-head). Renders well as a heatmap. |
| `concat(a, b)` | 2 | Concatenate two rank-0 or rank-1 arrays into a 1-D vector. Used in generation loops to append a sampled token id to the growing sequence. |
| `concat(a, b, axis)` | 3 | Axis-aware concat for any rank. Both inputs must agree on every dim except `axis` (sizes add); the forward accepts any `axis` in `[0, rank)`. Differentiable on the tape; the backward splits the upstream gradient at the seam. |
| `last_row(M)` | 1 | Return the last row of a rank-2 matrix as a rank-1 vector. Used in generation loops to extract the final position's logits from an `[T, V]` model output. |
| `patchify(x, P)` | 2 | ViT patch embedding rearrangement. Takes a `[B, C, H, W]` image batch and a square patch size `P` that divides both `H` and `W`. Returns `[B, N, P*P*C]` where `N = (H/P)*(W/P)` and each row of the trailing axis is one patch flattened in channel-outer order. Differentiable on the tape. |
| `shift_pairs_x(ids, block_size)` | 2 | Build next-token-prediction input windows from a 1-D token array. Returns an `[N, block_size]` integer matrix where each row is a contiguous window of `ids`. |
| `shift_pairs_y(ids, block_size)` | 2 | Matching target windows for `shift_pairs_x`: each row is the input window shifted right by one position. |
| `take(x, axis, idx)` | 3 | Drop one axis at a single integer index. Result has rank `rank(x) - 1`. Per-axis labels propagate (the dropped axis's label is removed). Differentiable on the tape: backward scatters the upstream gradient into a zero-filled array of the parent's shape at `axis = idx`. Multi-index gather and slice ranges are followups. |

### Embeddings and Manifold

| Function | Args | Description |
|----------|------|-------------|
| `knn(X, k)` | 2 | Return an `[N, k]` integer matrix of the `k` nearest non-self neighbors per row of `X`, sorted by ascending distance. Ties broken by lower original index. |
| `knn_graph(X, k)` | 2 | Return an `[N*k, 3]` edge list of the `k` nearest non-self neighbors per row of `X`. Each row is `(i, j, dist)` where `dist` is the Euclidean distance (not squared) from sample `i` to its `p`-th nearest neighbor. The explicit `i` column + distance makes this the input layer for UMAP and other graph-based dim-reduction methods. |
| `mds(X, k, iters, seed)` | 4 | Multidimensional Scaling: project `[N, D]` to `[N, k]` by minimizing the stress `sum_{i<j} (||Y_i - Y_j|| - d_ij)^2` between low-D coordinates `Y` and the input pairwise Euclidean distances `d_ij`. SGD with linear LR decay; deterministic given `seed`. Use when you want a projection that preserves PAIRWISE DISTANCES rather than variance directions (PCA) or local neighborhoods (t-SNE / UMAP). |
| `pairwise_sqdist(X)` | 1 | Return the `[N, N]` squared Euclidean distance matrix for an `[N, D]` input. Symmetric, zero diagonal. |
| `pca(X, k)` | 2 | Top-`k` principal-component projection of an `[N, D]` matrix via power iteration with deflation. Returns the centered, projected data `[N, k]` (not the components themselves). |
| `pca_components(X, k)` | 2 | Top-`k` principal-component LOADINGS of an `[N, D]` matrix. Returns `[k, D]` -- row `i` is the i-th principal-component direction in original feature space. Pairs with `svg(_, "critical_dimensions", names)` for per-feature importance heatmaps. |
| `pca_variance_explained(X, k)` | 2 | Returns a `[k]` vector of variance-explained ratios `lambda_i / trace(Cov)`. Sums to 1.0 when `k == D`. Useful as a legend on the loadings heatmap or as a stopping criterion for picking `k`. |
| `random_projection(X, k, seed)` | 3 | Johnson-Lindenstrauss random projection: multiply `[N, D]` by a seeded Gaussian random matrix `R [D, k]` scaled by `1/sqrt(k)`, giving `[N, k]`. For modest `k = O(log N / eps^2)` the JL lemma guarantees all pairwise distances are preserved within a `1 +- eps` factor. Useful as a sanity baseline against PCA / t-SNE / UMAP -- if a learned method does not beat random projection, the learned features are not adding value. |
| `tsne(X, perplexity, iters, seed)` | 4 | t-SNE 2D embedding of an `[N, D]` matrix. Returns `[N, 2]`. Deterministic for a given seed. Output has rotation / reflection ambiguity; cluster shape is what is meaningful, not absolute coordinates. |
| `umap(X, n_neighbors, min_dist, iters, seed)` | 5 | UMAP 2D embedding of an `[N, D]` matrix. Returns `[N, 2]`. Builds a k-NN graph (k = `n_neighbors`), computes the fuzzy simplicial set (per-row sigma calibration + symmetric fuzzy union), then optimizes the layout via SGD on a cross-entropy + repulsion objective with negative sampling. `min_dist` is a soft floor on attractive distances; smaller values pack clusters tighter. Deterministic given the same `seed`. Preserves both local neighborhoods (like t-SNE) AND global inter-cluster distances (unlike t-SNE) -- the recommended default for visualizing high-D embeddings. |

These primitives also carry the classical-ML story: the
"K-Nearest Neighbors" demo builds the cross-set distance matrix
from the `|a|^2 + |b|^2 - 2ab` identity (and checks it against
`pairwise_sqdist`), and "Naive Bayes (Gaussian)" fits a generative
classifier with two masked `matmul`s -- no training loop. See the
Classical ML demo category.

### Experiments

| Function | Args | Description |
|----------|------|-------------|
| `compare(name_a, name_b)` | 2 | Return a `Value::Str` with a side-by-side view of the most-recent runs with those names, including per-metric deltas. Merges memory-only (web REPL) and on-disk (terminal REPL, under `--exp-dir`) records. |
| `experiment_metric("name")` | 1 | One recorded metric across the in-memory experiment log, as a `[runs]` vector in run order. Runs that did not record the metric are skipped; an unrecorded metric yields the empty `[0]` vector. Column-concat several calls into the `[n, k]` matrix `pareto_front` eats. |
| `param_count(m)` | 1 | Total trainable parameters across the model's `param` arrays -- the size axis of a quality-vs-size frontier. Accepts a bound model name or an inline constructor. |

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
- `"gallery"` -- an `[N, 3, H, W]` image batch rendered as an SVG grid of RGB thumbnails. Values in `[-1, 1]` normalized space (clamps out-of-range). Thumbnails are downsampled via block averaging to keep the SVG size tractable for batches like the 20-image pets_tiny slice.
- `"decision_boundary"` -- a 2D classifier-output grid rendered as a diverging-color surface, with the third argument as an Nx3 `[x, y, label]` training matrix overlaid as colored points.

The browser REPL detects SVG return values and renders them inline.
The CLI REPL prints a `[svg: N bytes]` summary; pass `--svg-out <dir>`
to write each SVG to a file.

### Analysis helpers

High-level helpers that compute and render a complete diagram in
one call. Each returns an SVG string just like `svg()`.

| Function | Args | Description |
|----------|------|-------------|
| `boundary_2d(grid_outputs, dims, points, labels)` | 4 | Render a 2D classifier surface from a length-(rows*cols) vector and `[rows, cols]` dims, with separately-supplied training points and labels |
| `confusion_matrix(predicted, actual)` | 2 | KxK heatmap of class-id predictions vs actual labels with cell counts overlaid |
| `hist(data, bins)` | 2 | Histogram of a vector with `bins` equal-width bins, rendered as a bar chart |
| `loss_curve(losses)` | 1 | Vector of losses rendered as a line plot with axis labels |
| `loss_landscape(surface, dims, path)` | 3 | A `rows*cols` loss surface (dark = low loss) over `[rows, cols]` dims, overlaid with an optimizer trajectory `[N, 2]` of points normalized to `[0, 1]`; green dot = start, red = end |
| `scatter_labeled(points, labels)` | 2 | Nx2 points colored by a length-N cluster-id vector |
| `train_val_curve(train, val)` | 2 | Two loss vectors (training green, validation peach) on shared axes; the gap between them is overfitting |

## Scripting

Output primitives + Result-returning string conversions for
`mlpl-repl -f script.mlpl`. The output primitives return their
argument unchanged so they compose into expressions
(`x = print(some_computation)` both binds `x` and shows the value).
The string conversions return `Value::Result` so the caller branches
explicitly on failure via `is_ok` / `unwrap_or` / `err_message`.

| Function | Args | Description |
|----------|------|-------------|
| `print(v, ...)` | 1+ | Write each argument's display form to stdout, space-joined (println-style), then a newline. Returns the last argument (so `print(v)` still yields `v`). Variadic, so labelled output works without string concatenation: `print("count:", n)` prints `count: 3`. The display form matches what the REPL prints for each value's type. |
| `eprint(v, ...)` | 1+ | Same as `print` but writes to stderr. Useful for diagnostics that should not interleave with the script's main output stream. |
| `to_number(s)` | 1 | Parse `s` (a `Value::Str`) as an `f64`. Returns `Ok(scalar)` on success; `Err("to_number: cannot parse \"abc\" as a number")` on failure. Leading/trailing whitespace is trimmed. |
| `to_int(s)` | 1 | Parse `s` as an `i64`, rejecting non-integer numeric strings. Returns `Ok(scalar)`, `Err("to_int: \"3.5\" is not an integer")`, or `Err("to_int: cannot parse \"xyz\" as an integer")`. |
| `env(name)` | 1 | Read the OS environment variable `name`. Returns `Ok(string-value)` if set; `Err("env: NAME not set")` if unset. Pair with `unwrap_or(env(\"VAR\"), \"default\")` for a fallback. |
| `args()` | 0 | Return a `StrList` of the trailing CLI args passed to the script after the `--` separator (`mlpl-repl -f script.mlpl -- foo bar` makes this `["foo", "bar"]`). Empty list when run from the interactive REPL or the web playground. |
| `list_get(xs, i)` | 2 | Index into a `StrList` and return the `i`-th string wrapped in `Result`. `Ok(string)` when `i < len(xs)`; `Err("list_get: index N out of bounds (list has M items)")` when out of range. Pair with `unwrap_or(list_get(args(), 0), "default")` for a missing-arg fallback. |
| `read_stdin()` | 0 | Block until EOF and return all stdin bytes as a `Value::Str`. Refuses to read from an interactive terminal: `Err("read_stdin: stdin is a terminal; pipe input or use args() instead")` when stdin is a TTY. Pair with `print(read_stdin())` in shell pipes like `echo "hello" \| mlpl-repl -f greet.mlpl`. |
| `read_stdin_lines()` | 0 | Same EOF read as `read_stdin()` but split on `\n` and return a `StrList`. A trailing newline is stripped so `"a\nb\n"` and `"a\nb"` both yield `["a", "b"]`. Combine with `list_get`/`list_len` for line-oriented input processing. |
| `exit(code)` | 1 | Terminate the script with the given integer exit code. `code` must be in 0..=255; out-of-range or non-integer codes raise an eval error before the exit fires. `exit(0)` is clean; `exit(1)` is the usual "something went wrong" code. Never returns. |

### Script exit codes

In `mlpl-repl -f script.mlpl` mode, the process exit code is
determined by the script's final expression:

- Final value is `Err(msg)` -> exit code `1`, with `msg` written
  to stderr.
- Final value is `Ok(_)` or any non-`Result` value -> exit code `0`.
- Parse or eval error -> exit code `1`, with the source line and
  error written to stderr (existing behavior).
- `exit(code)` short-circuits all of the above and exits with the
  caller-chosen code.

This lets MLPL scripts compose with Unix tooling:

```sh
mlpl-repl -f check.mlpl && echo "ok"
mlpl-repl -f maybe-fail.mlpl || echo "failed: $?"
echo "1 2 3" | mlpl-repl -f sum-stdin.mlpl
```


### if / else expression

`if cond { then } else { else }` returns the value of whichever
branch is chosen. The `else` clause is required (no dangling-if).
Both bodies are statement sequences, with the final statement's
value being the branch value -- same convention as `repeat` /
`train` / `for` blocks.

`cond` is truthy when:

- It is a scalar `Number` and not zero (any non-zero value,
  including negatives, NaN, and infinity).
- It is a `Result` and its `ok` field is true (i.e. `Ok(_)`).

Other types (non-scalar arrays, strings, string lists, records,
etc.) raise a runtime error.

```mlpl
flag = 1
x = if flag { 100 } else { 200 }    # x = 100

# Result-as-condition: branches on the Ok / Err discriminant.
name = unwrap_or(env("USER"), "guest")
greeting = if env("USER") { "hello " + name } else { "no user" }

# Nested:
sign = if x { if x > 0 { 1 } else { -1 } } else { 0 }
```

### while / break / continue

`while cond { body }` re-evaluates `body` until `cond` is falsy
(same truthiness rule as `if`) or `break` fires. The expression
evaluates to:

- the `break value` if the loop exited via `break value`
- scalar `0` if the loop exited via bare `break`
- scalar `0` if the loop exited because `cond` went falsy

`continue` skips the rest of the current iteration; the condition
is re-checked. `break` and `continue` are only valid inside a
`while` body -- using either outside raises a runtime error.

```mlpl
# Count up to 5 (loop value is 0; i ends at 5).
i = 0
while i - 5 { i = i + 1 }

# Break with a value.
first_good = while 1 {
  x = next()
  if is_valid(x) { break x } else { 0 }
}

# Continue skips the s-update for i == 3.
i = 0
s = 0
while i - 5 {
  i = i + 1
  if i - 3 { 0 } else { continue }
  s = s + 1
}
```

Loops do not introduce a new scope: assignments in the body
persist in the surrounding environment, matching `repeat` /
`train` / `for` semantics.

### try / catch (error trap)

`try { body } catch e { handler }` is an EXPRESSION: it yields the
body's last value, or -- when the body raises a HARD error -- the
handler's, with `e` bound to the canonical error record
`{kind, message}` (dispatch on `e.kind`; kinds are stable kebab-case
tags like `shape`, `arity`, `runtime`). `err(...)` VALUES are data,
not hard errors: they flow through untouched. break/continue/return
signals also pass through. Design: docs/error-handling.md.

### `?` (Result propagation)

Postfix `?` (sugar for `check(expr)`): if the expression is `Ok(v)`,
continue with `v`; if `Err(e)`, EARLY-RETURN that whole Result from
the enclosing `def u:` function (the railway pattern). At top level
(no enclosing function) an `Err` is loud, like `unwrap`. Applying
`?` to a non-Result is an error.

## Array Display

Arrays are displayed in a row-major layout:

```
mlpl> 42
42
mlpl> [1, 2, 3]
1 2 3
mlpl> reshape(range(6), [2, 3])
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
