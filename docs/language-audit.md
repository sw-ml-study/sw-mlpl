# MLPL language audit + breaking-change candidates

Status: living document. Seeded as Saga 29 step 028. Update as new
sharp edges surface or as breaking changes land.

## Why this exists

MLPL is in alpha. Breaking changes are cheap now and expensive later.
This document captures every place the surface diverges from what an
experienced user of APL / J / BQN or PyTorch / JAX or Rust would
expect, classifies the divergence, and proposes a fix with migration
cost so the saga schedule can prioritize.

The reader is assumed familiar with at least one of the comparison
languages. The examples are concrete MLPL fragments; if a fragment
looks contrived, that is the point -- the contrivance is the audit
finding.

## Comparison languages -- what each contributes

- **APL / APL2 / J / BQN.** The array tradition. Strong on
  primitives that compose by rank (each, reduce, scan, outer
  product), strong on tacit programming (forks, hooks, trains, J's
  rank-conjunction), strong on consistent scalar / vector / matrix
  treatment via implicit broadcasting. Weak on named bindings as a
  first-class concept; the trade is what MLPL has chosen NOT to
  follow.
- **Python + PyTorch + JAX.** The current ML-research baseline.
  PyTorch contributes the autograd-tape mental model and the
  `nn.Module` parameter-binding pattern; JAX contributes
  `grad` / `vmap` / `jit` as first-class transforms, pure-function
  discipline, and the `pytree` shape-checking story. The combined
  ergonomics around shape errors and module composition are the
  obvious bar for MLPL.
- **Rust.** Strong types, exhaustive enum dispatch, no implicit
  coercions, error types that say which call site failed.
  MLPL inherits Rust's enum-and-trait dispatch internally; the
  question is how much of that discipline leaks (or should leak)
  to the surface language.

## How to read each finding

```
### N. <short title>

**Category:** MISSING | INCONSISTENT | ERROR-PRONE | ANTI-PATTERN | ALPHA-LEAK
**Priority:** critical | nice-to-have | cosmetic
**Where:** which crate / builtin / surface form

The issue, in one paragraph. Then a concrete MLPL fragment that
demonstrates the issue. Then the precedented alternative from one or
two of the comparison languages. Then the proposed breaking change.
Then migration cost (which demos / tests / docs touch).
```

Findings are grouped by category. Within each category they are
ordered roughly by priority (critical first).

---

## ERROR-PRONE (foot-guns)

### 1. Closures do not differentiate -- `train` requires inline forward

**Category:** ERROR-PRONE
**Priority:** critical
**Where:** `mlpl-eval/src/grad.rs`, `experiment { }` / `train { }`

The autograd tape walks the AST of the loss expression literally.
A `let` binding inside the loop that names the forward pass is
NOT followed -- the bound name becomes opaque to `grad`. So the
training loop must inline the forward expression verbatim:

```mlpl
# Does NOT learn: forward is captured by name, grad sees an opaque leaf
experiment {
  forward = apply(model, X)               # opaque to grad
  loss    = mse(forward, Y)
  step(loss)
}

# Learns: forward expression inlined
experiment {
  loss = mse(apply(model, X), Y)
  step(loss)
}
```

The footgun is silent -- the version with the `let` binding still
runs, the loss just plateaus.

**Precedent.** PyTorch is fine with named intermediates because
the tape is captured at op-execution time, not by AST walk:
`pred = model(x); loss = mse(pred, y); loss.backward()` works.
JAX's `grad(f)` requires `f` to be a function, which forces the
forward expression to be self-contained -- it cannot accidentally
escape into a name that grad will not follow.

**Proposed fix.** Switch the tape from AST-walking to value-capture:
when an autograd-tracked tensor enters the eval, every op records a
tape node; `grad` differentiates the tape, not the AST. Named
intermediates then flow through transparently. This is the largest
single change in the doc; it touches `mlpl-eval/src/grad.rs`,
`eval.rs`, and every demo's training block.

**Migration cost.** ~25 demos rewrite their `experiment` blocks to
the cleaner intermediate-binding form. The verbose inline forms keep
working (a no-op for them). Tests in `mlpl-eval/tests/grad_*.rs` and
`autograd_*.rs` exercise the new path; existing parity tests against
finite differences should still pass.

---

### 2. `device("mlx") { }` -- model params must be built INSIDE the scope

**Category:** ERROR-PRONE
**Priority:** critical
**Where:** `mlpl-eval/src/model_dispatch.rs`, `mlpl-mlx-rt`

A model constructed at top level lives on the CPU. Calling
`apply(model, X)` inside a `device("mlx") { }` block where `X` was
also constructed inside fails with `DeviceMismatch`:

```mlpl
# Fails at apply: linear_p is on CPU, X is on MLX
linear_p = linear(768, 128, 7)
device("mlx") {
  X = randn(0, [16, 768])
  Y = apply(linear_p, X)        # DeviceMismatch
}

# Works: model and inputs both built inside the scope
device("mlx") {
  linear_p = linear(768, 128, 7)
  X = randn(0, [16, 768])
  Y = apply(linear_p, X)
}
```

The error message names the mismatch correctly but the user has no
way to MIGRATE a CPU model to MLX -- only to construct one fresh
on MLX. New users hit this within the first MLX demo.

**Precedent.** PyTorch's `model.to("cuda")` migrates parameters in
place; JAX's `jax.device_put(pytree, device)` returns a relocated
copy. Both treat device residency as a property of the value, not a
property of the construction context.

**Proposed fix.** Add `to_device(target, value)` polymorphism for
model values (today only tensor values relocate). Implementation:
walk the parameter tree, `to_device` each leaf. Optional second
step: implicit relocation on `apply(model, X)` when one operand is
CPU and the other is MLX -- relocate the smaller operand. This is
debatable; it hides cost.

**Migration cost.** Several MLX demos can be simplified to construct
the model outside the `device("mlx") { }` scope. Internal call sites
of `model_dispatch` need to handle the relocation. No tests break.

---

### 3. Booleans encoded as `0.0` / `1.0` floats

**Category:** ERROR-PRONE
**Priority:** critical
**Where:** all comparison ops, `is_ok` / `is_err`, mask ops

Comparison operators and predicates return `f32` `0.0` / `1.0`. The
language has no `Bool` type. Truthiness on `if` / `where` is
`!= 0.0`, which means NaN is truthy and `1e-300` is truthy:

```mlpl
# These are indistinguishable as "truthy"
flag = 1.0
flag = 1e-300
flag = nan          # also truthy!
if flag { ... }
```

This is a deliberate APL inheritance (APL also uses 0/1). The
problem is that MLPL is NOT APL -- it has named bindings and Rust-
flavored errors, and users coming from Python or Rust expect a
distinct `Bool`. Bitwise tricks (`a & b` for boolean AND on `f32`)
don't work, since `&` is integer-bitwise on a float layout.

**Precedent.** PyTorch has a dedicated `torch.bool`; JAX's bool dtype
is a real dtype; NumPy distinguishes `bool` from `float32`. Rust
forbids implicit `bool -> integer` -- explicit `as u8` is required.

**Proposed fix.** Add `Value::Bool` (and a `bool` tensor dtype)
alongside the existing `Value::Number`. Promote comparison results
to `Bool`. Keep `0.0` / `1.0` interop one cast away (`to_float(b)` /
`to_bool(x, threshold)`). Update `is_ok` / `is_err` to return `Bool`.

**Migration cost.** Roughly 40 demos use comparison results; about a
quarter feed them into arithmetic (e.g., `correct = sum(pred == y)`)
and would need an explicit `to_float`. The `where` builtin's
predicate argument accepts both. Tests update straightforwardly.

---

### 4. Magic seed constants threaded through every layer constructor

**Category:** ERROR-PRONE
**Priority:** nice-to-have
**Where:** `linear`, `attention`, `embed`, `randn`, every dataset gen

Every random-using constructor takes a `seed` parameter explicitly:

```mlpl
linear_p   = linear(768, 128, 7)
attn       = attention(128, 4, 9)
classifier = chain(linear(128, 64, 11), relu_layer(), linear(64, 2, 13))
```

The numbers `7`, `9`, `11`, `13` mean "give me determinism" but
encode nothing. Users typically copy them from the nearest demo
without understanding -- a magic-number anti-pattern in disguise.
Worse, two layers with the same seed and same shape produce
identical weights, which silently breaks the symmetry-breaking
assumption neural networks need.

**Precedent.** PyTorch / JAX both use a global PRNG that is split
explicitly (JAX `random.split(key)`) or seeded once globally
(PyTorch `torch.manual_seed`). The user signs up for determinism at
the top of the program, not at every constructor.

**Proposed fix.** Two-tier API:

1. `seed(K)` at the top of the script seeds the global PRNG. All
   constructors default to splitting from this stream.
2. Constructors keep an optional explicit seed for reproducibility
   tests (named arg: `linear(in, out, seed=7)`), but it is opt-in.

**Migration cost.** Every demo file. Search-and-replace `linear(a,
b, NN)` -> `linear(a, b)` is mechanical and the script `seed(0)` is
prepended. Tests asserting specific weight values move from
positional seeds to keyword seeds.

---

### 5. `:upload` -- `Err("cancelled")` and `Err("decode failed")` are stringly-typed

**Category:** ERROR-PRONE
**Priority:** nice-to-have
**Where:** `apps/mlpl-web/src/state.rs`, `:upload` REPL command

A failed upload returns `Err("cancelled")`, `Err("decode failed: not
a valid image")`, `Err("read failed")`. The user has no way to
branch on the failure mode without string matching:

```mlpl
# Stringly-typed dispatch on the error
if err_message(x) == "cancelled" { ... }
```

**Precedent.** Rust's `io::ErrorKind`, Python's exception subclass
tree. Both let the program differentiate intent (user-cancelled vs
file-corrupt vs permission-denied) without parsing the message.

**Proposed fix.** Promote `:upload` errors to a sealed-tagged form:
`Err({ kind: "cancelled", message: "User dismissed the picker" })`,
`Err({ kind: "decode_failed", message: ... })`. Add `err_kind(r)`
alongside `err_message(r)`.

**Migration cost.** Tiny -- one builtin (`:upload`), the
`Value::Result` accessors gain `err_kind`, the bring-your-own-image
note in the Vision path mentions the kinds.

---

## INCONSISTENT (two builtins disagree)

### 6. `concat(a, b)` vs `concat(a, b, axis)`

**Category:** INCONSISTENT (also: arity-overloaded)
**Priority:** nice-to-have
**Where:** `mlpl-runtime/src/builtins.rs:205`, `math_builtins.rs:62`

`concat` accepts both 2-arg (defaults axis to 0) and 3-arg forms,
and the 3-arg form only supports `axis in {0, 1}`. Most other
builtins are strict arity. This is the first place a new user
discovers that MLPL has variadic builtins -- it should not be the
case that the first one they meet is also limited to two axes.

```mlpl
concat(a, b)          # axis 0 implied
concat(a, b, 0)       # axis 0
concat(a, b, 1)       # axis 1
concat(a, b, 2)       # ERROR: axis 2 not supported
```

**Precedent.** NumPy `np.concatenate([a, b], axis=N)` is variadic
over the LIST and supports any axis. JAX same. PyTorch
`torch.cat(tensors, dim=N)` same.

**Proposed fix.** Two changes:

1. Lift the axis restriction: arbitrary axis on rank-N inputs.
2. Promote `concat` to list-variadic: `concat([a, b, c, ...], axis)`
   matching the existing `stack` op. Keep the 2-arg `concat(a, b)`
   as sugar for `concat([a, b], 0)`.

**Migration cost.** Five demos use the 3-arg form; rewrite them
once. The new list-variadic form is also what `stack` accepts, so
the parallel structure is a learning aid.

---

### 7. Builtin naming -- some `verb_noun`, some `noun_verb`

**Category:** INCONSISTENT **Priority:** cosmetic

Names accreted across sagas without a rule. `verb_noun`:
`train_bpe`, `load_images`, `fetch_dataset`, `predict_batch`.
`noun_verb`: `cosine_schedule`, `linear_warmup`, `loss_curve`,
`attention_weights`. `verb` alone: `concat`, `take`, `patchify`,
`argmax`. `noun` alone: `softmax`, `tanh`, `relu`. Precedent:
NumPy / PyTorch mostly `verb`, `noun` for *return-the-noun* ops
(`mean`, `std`). Fix: ops named as their output; constructors as
nouns (`linear`, `attention`); side-effecting loaders as
`verb_noun`. Mechanical search-and-replace; defer until a bigger
refactor amortizes the demo-touch cost.

---

### 8. `svg(x, "heatmap")` -- diagram type as untyped string

**Category:** INCONSISTENT (also: error-prone)
**Priority:** nice-to-have
**Where:** `mlpl-runtime/src/grid_builtin.rs`, `svg` builtin

The third argument to `svg` is a string literal identifying the
visualization type: `"heatmap"`, `"heatmap_grid"`, `"gallery"`,
`"loss_curve"`, etc. There is no compile-time check; a typo like
`"heatmpa"` is a runtime error that surfaces only when that branch
of the demo runs:

```mlpl
svg(weights, "heatmpa")     # runtime error: unknown viz type
svg(weights, "heatmap")     # correct
```

The viz registry also accreted unevenly -- some types take 2 args
(`svg(x, "heatmap")`), some take 3 (`svg(X, "gallery", overlay)`),
some have hidden default parameters.

**Precedent.** Rust enums dispatched via match; the compiler refuses
to compile a misspelled variant. Even Python at least makes you
import the constructor, surfacing the typo at import time.

**Proposed fix.** Introduce a typed `Viz` namespace:
`svg(weights, Viz::Heatmap)`, `svg(X, Viz::Gallery(overlay))`. Each
variant carries the right arity. Keep string sugar for one
deprecation cycle.

**Migration cost.** Touches every `svg(...)` call in demos and tests
(~60 call sites). Mechanical. The string sugar lets the migration
be split across multiple commits.

---

### 9. Axis position is inconsistent across builtins

**Category:** INCONSISTENT **Priority:** cosmetic

`take(x, axis, idx)` -- axis second. `concat(a, b, axis)` -- axis
third. A user who knows `take` looks up `concat` to remember the
slot. Precedent: NumPy / PyTorch / JAX use `axis=` as a keyword
argument so position does not matter. Fix: keyword arguments at the
surface (`take(x, axis=1, idx=0)`, `concat([a, b], axis=1)`).
Internal builtins still positional. Small parser change; cleanest
in the same wave as fix #4. ~30 demo call sites update opt-in.

---

## MISSING (capabilities the language does not have)

### 10. No `vmap` / batched transform

**Category:** MISSING
**Priority:** critical (for the second wave of demos)
**Where:** entire language; `mlpl-eval`

There is no way to take a function written for one example and apply
it to a batch without re-writing it. The batched attention path was
added by hand-rewriting the autograd tape; the batched data path was
added by reshaping inputs to add a leading batch axis. There is no
`vmap(f)`.

**Precedent.** JAX `vmap` is the headline transform. PyTorch
`torch.func.vmap` exists. It changes which models are tractable to
write.

**Proposed fix.** Phase 1: `vmap(f, in_axes, out_axes)` as a builtin
that, when given an MLPL function value, rewrites the call to add a
batch axis at the named position. Phase 2: integrate with the tape
so gradients of vmapped functions work transparently.

**Migration cost.** Pure addition. Demos using
hand-batched-attention can stay as-is, but new demos pick up the
transform.

---

### 11. No `jit` / no compilation boundary

**Category:** MISSING
**Priority:** nice-to-have
**Where:** entire language

Every eval is a tree-walk through the AST. The interpreter loop
walks `mlpl-runtime`'s dispatch table per node; per-iteration cost
on the Tiny MLP demo's 600-step loop is dominated by op dispatch,
not floating-point work. The interpreter contract specifically
recommends `cargo test --release` for demo-heavy crates to mitigate
this.

**Precedent.** JAX `jit` compiles a function once and reuses the
compiled trace. PyTorch 2.0 added `torch.compile`. Numba is the
Python answer.

**Proposed fix.** Phase 1: `jit(f)` returns a value that wraps `f`
and trace-compiles on first call, caching by shape tuple. Phase 2:
lower the cached trace to MLX / CUDA. The MLPL compiler effort
(see `docs/compiling-mlpl.md`, `docs/compiler-guide.md`) is the
adjacent track here.

**Migration cost.** Pure addition. The benchmark suite under
`crates/mlpl-bench/` becomes the testbed for trace caching.

---

### 12. No `gather` / no slice ranges -- `take` is single-index only

**Category:** MISSING
**Priority:** critical
**Where:** `mlpl-runtime/src/builtins.rs:211`

`take(x, axis, idx)` extracts ONE index along ONE axis. There is no
way to pick `K` indices without `K` calls and a stack. There is no
range slice `x[a..b]`.

```mlpl
# Three calls + stack just to pick 3 rows
rows = stack([take(x, 0, 0), take(x, 0, 2), take(x, 0, 5)], 0)
```

**Precedent.** NumPy `x[indices]` with fancy indexing; PyTorch
`torch.gather`; JAX `jnp.take` accepts a list of indices.

**Proposed fix.** Two additions:

1. `gather(x, axis, idx)` where `idx` is a rank-1 integer tensor.
2. Slice syntax: `x[1..5, :]` parses to `slice(x, 0, 1, 5)` (or
   similar). Surface this in the parser.

**Migration cost.** Pure addition. Patchify and the multi-head
attention reshape dance both simplify once `gather` lands.

---

### 13. No tacit / point-free programming

**Category:** MISSING **Priority:** cosmetic **Where:** parser

MLPL has no `fork` / `hook` / `train` from APL/J/BQN. "Subtract the
mean" is `\x. x - mean(x)`, not the APL `(- mean)`. This is a
deliberate non-choice -- MLPL has named bindings as the dominant
idiom -- but the audit notes it because the array-tradition reader
may misread the absence as an oversight. Defer; if added, scope to
forks only (not the full train algebra). Pure addition; no
migration.

---

### 14. No named-axis types

**Category:** MISSING **Priority:** nice-to-have **Where:** shape system

Shapes are anonymous tuples of `usize`. A `[B, T, D]` and a `[B, H,
W]` are indistinguishable to the type system; misaligning them
produces a silent shape error at a downstream broadcast.
`docs/milestone-named-axes.md` is the queued saga. Precedent:
einops `Rearrange("b h w c -> b c h w")`; PyTorch named tensors;
JAX + xarray. Audit's job here is to flag that "shape errors are
positional" is felt by every user writing a non-trivial reshape.

---

## ANTI-PATTERN (the language pushes you toward bad style)

### 15. Inline forward expression in `experiment { }` blocks

**Category:** ANTI-PATTERN (the root cause is finding #1)
**Priority:** critical
**Where:** every training demo

Because closures don't differentiate, every demo training loop has
to inline the forward expression even when the model has 6+ stages:

```mlpl
experiment {
  loss = cross_entropy(
    apply(classifier,
      take(
        apply(attn,
          reshape(
            apply(linear_p, reshape(patchify(X, 16), [16, 768])),
            [1, 16, 128])),
        1, 0)),
    Y)
  step(loss)
}
```

This is unreadable. The user CANNOT factor out the forward pass
without breaking grad (finding #1). New users copy this shape from
existing demos and propagate the smell.

**Precedent.** PyTorch and JAX both allow named intermediates in the
forward pass.

**Proposed fix.** Solve finding #1; the anti-pattern dissolves.
No separate fix needed.

**Migration cost.** Same as #1.

---

### 16. The "build all parameters at the top, then pass them everywhere" pattern

**Category:** ANTI-PATTERN
**Priority:** nice-to-have
**Where:** every multi-layer demo

Because there is no `Module` / `Layer` aggregating type, demos build
N separate `linear` / `attention` / `embed` parameter blobs and
thread them through the forward expression by name. With 8 stages
this becomes 8 explicit variables that must be remembered.

```mlpl
linear_p   = linear(768, 128, 7)
attn       = attention(128, 4, 9)
proj       = linear(128, 64, 11)
classifier = linear(64, 2, 13)
# ... 4 more
forward = apply(classifier, apply(proj, take(apply(attn, ... ), 1, 0)))
```

**Precedent.** PyTorch `nn.Module` aggregates params; JAX
`flax.linen.Module` and `equinox.Module` do the same.

**Proposed fix.** `model = chain(linear(768, 128), attention(128,
4), take(_, 1, 0), linear(128, 64), relu, linear(64, 2))` -- the
chain is the parameter aggregate. The model-DSL primitives
(`chain`, `residual`) already do this for the layers they cover;
the missing piece is bringing `take` and friends into the DSL so
the chain can be the WHOLE forward path, not just the linear /
attention parts.

**Migration cost.** Add DSL wrappers for `take`, `reshape`,
`patchify`, `softmax`. Rewrite ~20 demos to use the chain form.
The wrapped forms type-check as model fragments and compose.

---

### 17. Stringly-typed device names

**Category:** ANTI-PATTERN **Priority:** cosmetic

`device("cpu")` / `device("mlx")` are string literals. Typos
produce runtime errors at the routing layer. No compile-time
signal that `"cuda"` is not yet supported. Fix: `Device::Cpu`,
`Device::Mlx` typed namespace. Sugar the string form for a
deprecation window. Touches `device.rs` plus MLX demos.

---

## ALPHA-LEAK (works today only because nothing exercises it)

### 18. `concat` axis restricted to `{0, 1}`

**Category:** ALPHA-LEAK
**Priority:** critical (it has already been hit)
**Where:** `mlpl-array/src/ops.rs:469`

The concat implementation literally returns a shape error if `axis
> 1`. The user-visible error is `ShapeMismatch { source: 2, target:
1 }` -- which reads as "you passed mismatched shapes" when the real
error is "concat does not support axis 2 yet". This was tripped
during the ViT step where joining rank-3 batched-attention outputs
along the batch axis required a workaround.

**Precedent.** NumPy / PyTorch / JAX all support arbitrary axis
since day one.

**Proposed fix.** Implement axis-N concat in `mlpl-array`. The
existing `copy_concat_rows` helper generalizes by computing
contiguous strides.

**Migration cost.** Drop the workaround in `mlpl-eval`'s
attention-stack lowering. No demos rewrite.

---

### 19. `attention(d, h)` -- tape lowering only for `h = 1`

**Category:** ALPHA-LEAK
**Priority:** critical
**Where:** `mlpl-eval/src/grad.rs`, `attention` builtin

Single-head attention has full forward + backward through the tape.
Multi-head attention has forward only. Training a multi-head ViT
runs but the gradient is implicitly zero on the per-head splits --
the visible symptom is "loss drops a bit, then plateaus". The
quick-demo workarounds use the single-head autograd path with a
manual `apply` over heads.

**Precedent.** Multi-head attention is one of the headline tape
operations in any autograd framework.

**Proposed fix.** Lower multi-head attention onto the same tape
primitives (Q/K/V projection, per-head SDPA, stack, output proj)
that single-head uses. Saga 29 step 008 partially did this.

**Migration cost.** None for users; pure capability lift. The
multi-head ViT demo trains end-to-end after this.

---

### 20. The dispatch table in `mlpl-runtime` has implicit ordering

**Category:** ALPHA-LEAK **Priority:** cosmetic **Where:** `BUILTINS`

`BUILTINS` slice order determines which overload wins on a name
collision. Today no two collide; nothing prevents future addition
from silently shadowing. Fix: `debug_assert!` over the slice at
startup to reject duplicates. One-line guard; no migration.

---

### 21. `sw-checklist` 7-fn / 7-module budget shapes the code

**Category:** ALPHA-LEAK (process, not language) **Priority:** cosmetic

The 7-fn / 7-module budget has produced many small files
(`block_acc.rs`, `inline_render.rs`, `upload_cmd.rs`) where the
natural shape was one or two files. Sometimes a clarity win,
sometimes a loss. Flagged because the budget leaks into the code
shape users see in `crates/mlpl-eval/src/`. Out of scope for this
audit; note for the saga retrospective.

---

## Scripting / control-flow gap (added 2026-05-20)

A user asked whether MLPL could be used as a scripting language --
take command-line arguments and execute different paths through the
code. Today the answer is no on both halves. The next nine findings
make it yes. They share a theme: MLPL has been a *demo language*
where the script writer always knew the data and the path in
advance. Treating it as a *scripting language* (data unknown at
write time, paths chosen at run time) needs a small set of language
additions.

### 22. No surface `if` / `else` -- conditionals via arithmetic masks

**Category:** MISSING **Priority:** critical

Comparison builtins (`eq`, `gt`, `lt`) return `0.0` / `1.0`
floats; conditional logic is expressed as multiplication by a
mask. There is no `if cond { then } else { else }` in the parser
AST (`crates/mlpl-parser/src/ast.rs`). There is no ternary, no
`cond`, no `where`, no `select`. The Result accessors
(`unwrap_or(r, default)`) cover the Result branch case but
nothing else.

```mlpl
# How a "branch" looks today: mask + multiply
mask     = gt(x, 0.0)
positive = x * mask
negative = x * (1.0 - mask)
total    = positive + negative * 0.5
```

For a tensor this is fine and even preferred (no branching = no
warp divergence on a GPU). For a scalar command-line flag it is
absurd. A user wanting to write `if --train { train_model } else
{ predict_only }` has no surface.

**Precedent.** Every comparison language has `if`. APL has the
inline `:If` plus dyadic `?:`-equivalents; BQN has `f.{cond ? then ; else}`;
Python `if/elif/else`; Rust `if` as an expression.

**Proposed fix.** Add `if cond { then } else { else }` to the
parser as an expression (returns the chosen branch's value).
`cond` is truthy iff non-zero (matches the existing convention)
or, post-#3, a real `Bool`. Optional `elif`. The tensor case
stays one mask-multiply.

**Migration cost.** Pure addition. ~10 demos that pre-compute
masks could be rewritten more readably; not load-bearing.

---

### 23. No `while` loop; `repeat N { }` is fixed-count only

**Category:** MISSING **Priority:** nice-to-have

`repeat N { body }`, `train N { body }`, and `for x in source { body }`
are the loop forms (docs/lang-reference.md:132-181). All three need
the bound to be known up front. There is no `while cond { body }`,
no `loop { ... break }`, no early `break` / `continue`.

Consequence: a script that wants to "train until validation loss
stops dropping" or "keep generating tokens until the model emits
EOS" cannot. The `train N { }` block has to over-estimate `N` and
the user reads the loss curve by hand.

**Precedent.** Every imperative language. JAX has `lax.while_loop`
(with explicit carry state) because batching wants a static loop
structure; MLPL is already a tree-walker, so an eager `while` is
trivial.

**Proposed fix.** Add `while cond { body }` to the parser; body
value is the last iteration's last expression, like `repeat`.
Add `break` and `break value` keywords usable inside any loop
form (including `repeat` and `train`); `break value` makes the
whole loop return `value`. Also add `continue`.

**Migration cost.** Pure addition. The `train { }` block can keep
its semantics; users opt in to `while` for adaptive training.

---

### 24. No command-line argument capture in script mode

**Category:** MISSING **Priority:** critical

`mlpl-repl -f script.mlpl arg1 arg2 ...` parses the `-f` flag and
*silently drops the trailing args* (`apps/mlpl-repl/src/main.rs:17-48`).
There is no `args()` builtin, no `ARGV`, no positional-binding
syntax. A user trying to write a script that classifies an image
named on the command line has no path.

```sh
# Today: this is just "run the script", trailing args are lost.
mlpl-repl -f classify.mlpl my_cat.jpg
```

**Precedent.** Python `sys.argv`; Bash `$1 $2`; Rust
`std::env::args()`; Julia `ARGS`.

**Proposed fix.** Three small additions:

1. `mlpl-repl -f script.mlpl -- arg1 arg2` passes trailing args
   through (the `--` separator is conventional).
2. A new builtin `args()` returns a string list (`Value::StrList`,
   which already exists from Saga 29 step 002) of the trailing
   args. Empty list when run from the REPL with no args.
3. A new builtin `arg(i)` is sugar for `take(args(), 0, i)`.

**Migration cost.** Pure addition; the REPL keeps working without
the `--`. Two new builtins + a small CLI parser change.

---

### 25. No environment-variable access

**Category:** MISSING **Priority:** nice-to-have

`env("MODEL_PATH")` does not exist. A script that wants to read
the path to a fine-tuned model from `$MODEL_PATH` has no surface.

**Precedent.** `os.environ` (Python), `std::env::var` (Rust),
`Sys.getenv` (OCaml), `$ENV{X}` (Perl).

**Proposed fix.** `env(name)` builtin returning `Value::Result` --
`Ok(Str)` if set, `Err(Str)` if unset. The Result discipline
matches `:upload` and forces the script writer to handle the
missing-env case.

**Migration cost.** Pure addition. One builtin in
`crates/mlpl-runtime/src/builtins.rs`.

---

### 26. No string-to-number parsing

**Category:** MISSING **Priority:** critical (blocks #24 alone)

If `args()` returns a `StrList`, the script has to parse the
strings before it can use them as scalars. MLPL has no
`parse_int`, `to_number`, `to_f64`. A user typing
`mlpl-repl -f train.mlpl -- 100 0.001` cannot turn `"100"` and
`"0.001"` into the epoch count and learning rate.

**Precedent.** Every language has this. Python `int(s)`, `float(s)`.
Rust `s.parse::<f64>()`.

**Proposed fix.** Two builtins: `to_number(s)` returns
`Value::Result<Number, Str>` -- `Ok(n)` on success, `Err(msg)`
on parse failure (e.g. `"abc"`). `to_int(s)` similar for
integers. Both Result-typed for the same reason as `env`.

**Migration cost.** Pure addition. Two builtins.

---

### 27. No stdin reading

**Category:** MISSING **Priority:** nice-to-have

`mlpl-repl -f filter.mlpl < data.txt` discards stdin. A script
that wants to act as a Unix pipe filter has no input.

**Precedent.** `sys.stdin.read()` (Python), `read_to_string`
(Rust), `getline` (awk).

**Proposed fix.** `read_stdin()` returns `Value::Str` (the
whole stdin contents). `read_stdin_lines()` returns
`Value::StrList`. Both block until EOF.

**Migration cost.** Pure addition. Conditional on whether
stdin is a TTY -- a TTY read should not hang the REPL.

---

### 28. No `print` / explicit script output

**Category:** MISSING **Priority:** critical (blocks scripts alone)

A script's value is whatever its last expression returned;
intermediate values are silently dropped. In the REPL the user
sees them because the REPL prints each statement's result. In
`-f` script mode only the final value is displayed
(`apps/mlpl-repl/src/main.rs:87-113`). A script that wants to
print "classifying cat... done" between two computations cannot.

**Precedent.** `print` / `println` is universal. Rust
`println!`, Python `print`, AWK `print`.

**Proposed fix.** `print(value)` builtin: writes the value's
display form (the same format the REPL uses) to stdout, plus
newline. `eprint(value)` writes to stderr. Both return their
argument unchanged so they compose into expressions:
`x = print(some_computation)` binds and shows. A `print` that
returns unit would break the expression-only language model.

**Migration cost.** Pure addition. Two builtins.

---

### 29. No script exit code / error propagation

**Category:** MISSING **Priority:** nice-to-have

If a script encounters an error (an `Err(...)` Result, a parse
failure, a tensor shape mismatch), the REPL prints the diagnostic
and exits zero. There is no way for the script to communicate
"this failed" to a calling shell. Unix scripting depends on
exit codes.

**Precedent.** `sys.exit(n)` (Python), `exit(n)` (C/Rust),
`return n` from `main` (Bash).

**Proposed fix.** Two pieces:

1. If the script's final expression is `Err(msg)`, exit non-zero
   and print `msg` to stderr.
2. `exit(code)` builtin terminates immediately with the given
   integer code.

**Migration cost.** Pure addition. Light change in `mlpl-repl`'s
`run_script` to inspect the final value.

---

### 30. The "script" mental model is unsupported by example demos

**Category:** ANTI-PATTERN **Priority:** nice-to-have

Every demo under `demos/*.mlpl` hard-codes its inputs. None take
arguments, none branch on user choice, none read stdin or
files-by-name. The implicit message to a new user is "MLPL is a
notebook language, not a script language." The findings above
unlock script use; the demo set should follow with at least one
example that takes args and branches.

**Precedent.** Most language standard libraries ship at least
one "command-line tool" example.

**Proposed fix.** Add a `demos/classify.mlpl` once #22, #24, #26,
#28 land: takes a path argument, loads the image, picks a model
(passed via `--model` or `$MODEL`), runs inference, prints the
label and confidence to stdout. Exit code reflects success.
Document the script pattern in `docs/usage.md`.

**Migration cost.** None for existing surface; one new demo file
+ a section in the usage guide.

---

## Cross-cutting observations

- The single biggest user-visible win is **fixing
  closures-don't-differentiate (#1)**. Most other ergonomic
  complaints (#15, #16) are downstream of it.
- The next biggest is **booleans (#3)**. It is everywhere and the
  fix is mechanical.
- **`vmap` (#10) + `gather` (#12)** are the two missing primitives
  that show up most often when reading another author's demo and
  thinking "why is this so verbose."
- **Magic seeds (#4) + named axes (saga 19) + typed viz (#8)** are
  the three "MLPL accreted, time to consolidate" entries. They are
  not blockers but they are the kinds of fix that get harder at
  beta.

## Priority bucket for saga consideration

Critical (recommend pre-v1.0):

- #1 closures-don't-differentiate
- #2 device("mlx") param relocation
- #3 booleans-as-floats
- #10 vmap
- #12 gather + slice ranges
- #15 (downstream of #1)
- #18 concat axis-N
- #19 multi-head attention tape
- **Scripting cluster:** #22 `if`/`else`, #24 args, #26
  string-to-number, #28 `print`. These four are the minimum
  set for MLPL to function as a script language; landing any
  three without the fourth still leaves users blocked. Treat
  them as one saga.

Nice-to-have:

- #4 magic seeds -> global PRNG
- #5 :upload error kinds
- #6 concat list-variadic
- #8 typed viz dispatch
- #9 keyword args for axis
- #11 jit
- #14 named-axis types (saga 19)
- #16 model DSL coverage
- #17 typed device names
- #23 `while` + `break` / `continue`
- #25 env-var access
- #27 stdin
- #29 script exit code
- #30 example-demo coverage

Cosmetic:

- #7 builtin naming convention
- #13 tacit programming
- #20 BUILTINS duplicate guard
- #21 sw-checklist policy leak

## Maintenance

- Add a new finding when a user hits an unexpected sharp edge OR
  when adding a new builtin reveals a precedented alternative
  MLPL is not following.
- Move a finding to a fix-landed-here line (with commit hash) when
  the breaking change ships.
- When the priority bucket is empty at the critical tier, MLPL is
  ready for v1.0 from a language-design standpoint.
