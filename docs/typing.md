# MLPL Typing

**Status:** as of v0.7.0, MLPL is dynamically and weakly typed.
This document describes how typing works today, what is *not*
checked, and how to think about types when writing or reasoning
about MLPL programs.

For *future* directions (named axes, shape labels, typed ML
concepts like layers and weights), see:

- `docs/milestone-named-axes.md` -- Saga 11.5
- `docs/typed-ml-concepts.md` -- proposed typed ML surface
- `docs/are-we-driven-yet.md` -- agent-driven typing wishlist

## One-line characterization

> APL2-style dynamic untyped homogeneous arrays. Every numeric
> value lives in an `f64` dense buffer; shape mismatches are
> runtime errors raised by the op that detected them; there is
> no dtype, no static type checking, and no first-class function
> type.

## The three runtime value kinds

`Value` in `crates/mlpl-eval/src/value.rs` is a tagged union with
exactly three variants:

| Variant       | Holds                                  | Created by                                 |
|---------------|----------------------------------------|--------------------------------------------|
| `Array`       | a `DenseArray` (shape + `f64` buffer)  | every literal, every op, most built-ins   |
| `Model`       | a `ModelSpec` (Saga 11 layer tree)     | `linear`, `chain`, `residual`, `attention` |
| `Str`         | a string                                | string literals like `"heatmap"`           |

That is the entire universe of things a binding can hold. There
are no tuples, no records, no enums, no nullable / option, no
integer-vs-float distinction at the value level, no maps, and
no first-class functions.

## No dtype

Every numeric value lives in an `f64` buffer. There is no `int`,
`float32`, `bool`, `complex`, or `bfloat16`. Consequences:

- Booleans from `gt` / `lt` / `eq` are the floats `0.0` and `1.0`.
  Filtering and masking is done by elementwise multiplication.
- Integer indices passed to `iota`, `reshape`, `one_hot`, etc.
  are checked at the call boundary by demanding a scalar with
  zero fractional part, then stored as `f64` like everything else.
- Mixed-precision training, accumulator types, and quantized
  weights are all out of scope until a future saga adds dtype.

## Shape is the only structural type information

A `DenseArray` carries a `Shape`, which is a `Vec<usize>` of dim
sizes. Shape mismatches are runtime `EvalError` values raised by
the op that detected them:

- `matmul(a, b)` checks that `a.shape().dims()[-1] == b.shape().dims()[0]`
  (or rank-1 dot product equivalents) and otherwise raises
  `EvalError::Unsupported(..)`.
- Elementwise ops broadcast scalar-against-array but raise on
  incompatible array shapes.
- Reductions check the axis index is in bounds.

Nothing rejects an ill-shaped program before it runs. There is
no shape inference, no shape-polymorphic compilation, no
ahead-of-time validation pass. The first call that fails is the
one that crashes.

Saga 11.5 adds *labeled* axes (axis names alongside the dim
sizes) which improve error messages and enable
`:describe` to print `[seq=6, d_model=4]` instead of `[6, 4]`.
**Labels are metadata, not types.** Programs with labeled axes
still fail at runtime; the labels just make the failure
actionable.

## No type annotations, no inference, no generics

You can't write `x : Tensor[B, T, D]` today. The parser has no
colon-annotation syntax, no `let`-binding form, no type
parameters, and no traits or interfaces. Saga 11.5 adds an
optional annotation `x : [batch, time, dim] = ...` but that
attaches axis labels to a runtime value, not a static type.

## Polymorphism comes from ops, not values

Why does `[1, 2, 3] + 10` work? Because `mlpl-array`'s elementwise
primitive accepts both `(scalar, array)` and `(array, array)`
arguments and dispatches internally. There is no overloadable
trait for `+`; there is just one Rust function that handles every
broadcasting shape it knows. The same is true of `matmul` (which
handles vector-vector, vector-matrix, matrix-vector, and
matrix-matrix internally) and of every other built-in.

## Models look almost like functions, but aren't

`Value::Model` is the closest thing to a callable user-defined
thing in MLPL today. A model wraps a `ModelSpec` (a tree of
`Linear`, `Chain`, `Activation`, `Residual`, `RmsNorm`, `Attention`
nodes) and its trainable parameters. You "call" it with the
`apply(model_ident, X)` built-in.

But a model is not a function in any first-class sense:

- You cannot pass a model as an argument to another model.
- You cannot store models in an array.
- You cannot return a model from a built-in or from a `repeat`
  body (the `repeat` body returns a scalar).
- You cannot generalize over a model's arity -- every model takes
  exactly one rank-2 tensor input and returns one rank-2 tensor.
- You cannot put `if` / `else` / branching logic inside a model.
- You cannot recurse.

Models cover "parameterized neural-net forward passes" and
nothing else. They are data shaped to look like a callable, not
a callable in the language-design sense.

## No user-defined functions

There is no `def`, `fn`, `lambda`, dfn `{ ... }`, or `\` in MLPL
today. Every callable name resolves to a Rust-implemented
built-in registered in `mlpl-runtime` or `mlpl-eval`. Adding
user functions is a planned future addition (see
`docs/typed-ml-concepts.md` for the design sketch).

This means every higher-order construct that other languages
take for granted -- `grad(f, w)` taking a function, `vmap(f, x)`,
`map(f, xs)`, callbacks, etc. -- is unavailable. `grad` takes a
*loss expression* instead, and the expression is re-evaluated by
the autograd tape on every call.

## Comparison with related languages

| Language        | Typing model                                                                  |
|-----------------|-------------------------------------------------------------------------------|
| **MLPL v0.7.0** | dynamic, untyped, single homogeneous `f64` buffer, runtime shape checks       |
| APL2            | dynamic, untyped, mixed homogeneous arrays                                    |
| BQN             | dynamic, untyped, with array-of-array nesting and first-class functions       |
| J               | dynamic, untyped, with rank-polymorphic verbs                                 |
| NumPy           | dynamic, dtyped (`int32`, `float64`, ...), shape checks at op time            |
| PyTorch         | dynamic, dtyped + device-tagged, eager + traceable                            |
| MLX             | dynamic, dtyped + device-tagged, lazy graph                                   |
| JAX             | dynamic, dtyped + device-tagged + shape-traced under `jit`                    |
| Futhark         | static, dtyped, shape-polymorphic                                             |
| Dex             | static, dependent shapes, true types for tensors                              |

MLPL sits at the far "untyped + homogeneous" end of the spectrum,
deliberately, to keep the surface small while the language design
matures. Static typing, dtype, and shape-polymorphic compilation
are all candidate future sagas.

## Practical implications

Things you can rely on today:

- Every numeric expression evaluates to either a scalar or a
  rank-N `f64` array, depending on the op.
- Built-in functions raise `EvalError` early when the shape
  doesn't fit, with a string message naming the offending op.
- The `:describe <name>` REPL command will tell you the runtime
  shape of any binding, plus a values preview.
- The `:vars` REPL command lists every bound array with its
  shape, tagging trainable parameters with `[param]`.

Things you cannot rely on:

- That a program will type-check before running. It won't; it
  will run until it can't.
- That the dtype of a value is preserved across an op. There is
  no dtype to preserve.
- That a built-in's signature is documented anywhere except in
  its `:describe` entry and the source.
- That two values with the same numeric shape have the same
  semantic meaning. Saga 11.5 begins to fix this with labels.

## See also

- `docs/are-we-driven-yet.md` -- the typing wishlist from the
  agent-driven and paper-driven design discussions.
- `docs/milestone-named-axes.md` -- the next concrete step toward
  catching shape errors before they crash a run.
- `docs/typed-ml-concepts.md` -- the longer-range proposal to
  introduce typed ML concepts (weights, layers, hidden layers,
  activations) on top of the array substrate.
