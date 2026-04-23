# sw-checklist patterns

> Companion to `sw-checklist` itself. Budgets aren't
> arbitrary; they're a forcing function. This doc names
> the patterns that keep code under the budgets without
> stretching for lint appeasement.

## Why the budgets exist

`sw-checklist` enforces:

- **50 LOC/function** -- one screen, one mental chunk.
- **7 functions/module** -- one coherent responsibility
  per file.
- **500 LOC/file** -- forces module extraction before a
  file becomes a god-module.
- **7 modules/crate** -- keeps crate surface narrow;
  prevents "grab bag" crates that accumulate unrelated
  work.
- **No `#[allow(clippy::too_many_arguments)]`** and other
  allow-suppressions -- those are always a design smell
  that a struct was missing.

These are not lint taxes. They're design constraints. A
function that fits in 50 LOC is readable without
scrolling. A module with 7 well-named functions has a
clear shape. When a budget wants to fire, the refactor it
suggests almost always produces cleaner code.

**Design for the budgets up front.** Writing a 70-line
function and then extracting helpers works, but the first
draft reflects unfocused thinking. Identifying the
sub-responsibilities BEFORE typing (and naming the
helpers) produces a tighter first draft and a clearer
mental model.

## Pre-implementation checklist

Run through this before writing any non-trivial function:

- [ ] **Single responsibility.** What is this function
  for, in one sentence? If the sentence has "and" in it,
  it's two functions.
- [ ] **Inputs.** More than 4 positional args? Pack into
  a named struct. Named fields read better at the call
  site and survive argument-order changes.
- [ ] **Outputs.** More than 3 return values? Return a
  struct. Tuples are fine for 2-3; past that, field
  names carry meaning.
- [ ] **Body size estimate.** If you can already see >50
  lines of work, identify the sub-responsibilities NOW
  and name the helpers before typing.
- [ ] **State mutation.** Can the mutation be scoped to a
  narrow helper? Can some of the work be pure functions
  that take data in and return data out?
- [ ] **Module fit.** Does the function belong in this
  module's responsibility? If the module name no longer
  describes what lives in it, extract.

## Pattern catalog

Each pattern includes a one-line rationale and a
real-world example from this repo.

### 1. Struct return (> 3 return values)

**When:** a function naturally wants to hand back 4+
values. A tuple is opaque at the call site; named fields
carry meaning.

**Example (Saga 16 step 002):**

```rust
// BAD: clippy complains; call site unreadable.
fn validate_tsne_args(...)
    -> Result<(usize, usize, f64, usize, f64, Vec<f64>), RuntimeError>

// GOOD: named fields.
struct TsneArgs {
    n: usize, d: usize, perplexity: f64, iters: usize,
    seed: f64, xs: Vec<f64>,
}
fn validate_tsne_args(...) -> Result<TsneArgs, RuntimeError>
```

### 2. Struct args (> 4-5 params)

**When:** function signature is getting wide. Bonus:
struct fields are optional in call sites (via `..`), so
adding a new param doesn't ripple.

**Example (Saga 15 step 002):** `LoraCtx { rank, alpha,
seed, counter }` and `AdapterInit { in_dim, out_dim,
rank, device, a_seed }` turned what would have been
6-arg and 5-arg helpers into 2-arg `&LoraCtx` /
`&AdapterInit` calls.

### 3. Validate-then-work split

**When:** a function body is 40% input validation + 40%
actual work. Split into `validate_X(raw) -> Ok(clean)`
and a worker that takes the already-validated input.

**Example (Saga 16 step 001):** `validate_knn_args`
extracts all precondition checks and returns `(n, d,
k, xs)`, leaving `builtin_knn` at ~25 LOC of actual
work.

### 4. Orchestrator + helpers (facade pattern)

**When:** a function has a natural "phase 1, phase 2,
phase 3" structure.

**Example (Saga 15 step 003 -- LoRA forward):**

```rust
fn apply_linear_lora(x, inputs, env) {
    // Orchestrator; reads inputs from the struct, calls
    // each phase. 25 LOC.
    let xw = compute_base_wx(x, inputs, env)?;
    let xab_scaled = compute_adapter_delta(x, inputs, env)?;
    let b_broadcast = compute_bias_broadcast(xw.shape(), inputs.b, env)?;
    sum_three_terms(xw, xab_scaled, b_broadcast, env)
}
```

Each helper has a single named job. The orchestrator is a
readable story.

### 5. Per-variant helpers (extract match arm bodies)

**When:** a match on an enum has 5+ variants and any arm
body is >5 lines.

**Example (Saga 15 step 002 -- `clone_spec`):**
originally one match with 7 inline arm bodies totaling
70 LOC. Split into `clone_linear(env, w, b)`,
`clone_embedding(env, spec)`, `clone_attention(env,
spec)`, `clone_linear_lora(env, spec)` -- each ~10 LOC
and independently testable. The outer match became a
pure dispatch.

### 6. Phase helpers (split a linear algorithm)

**When:** an algorithm has distinct sequential phases.
Each phase becomes a pure function taking the previous
phase's output.

**Example (t-SNE decomposition):**

```rust
fn builtin_tsne(args) -> Result<DenseArray, RuntimeError> {
    let args = validate_tsne_args(args)?;
    let d2 = compute_pairwise_sqdist(&args.xs, args.n, args.d);
    let p_cond = bisect_perplexity_per_row(&d2, args.n, args.perplexity);
    let p = symmetrize_and_clamp(p_cond, args.n);
    let mut y = init_y(args.n, args.seed);
    let mut update = vec![0.0; args.n * 2];
    for step in 0..args.iters {
        tsne_gd_step(&mut y, &p, &mut update, args.n, schedule(step));
    }
    Ok(DenseArray::new(Shape::new(vec![args.n, 2]), y)?)
}
```

Each phase is a pure function; the orchestrator is
short and readable; each phase is trivially unit-testable
because its inputs and outputs are explicit.

### 7. Extract to new module

**When:** 3-4 related functions want to live together,
and they'd double a module's responsibility.

**Example:** Saga 15's `model_freeze.rs`,
`model_lora.rs`, `model_clone.rs` all extracted from
`model_dispatch.rs` when they would have pushed it past
its budget. Each new module has a clear one-line
purpose.

### 8. Chain-of-responsibility dispatch

**When:** you're matching on a name against a growing
list of patterns.

**Example (`mlpl-runtime`):** each builtin module
exports `try_call(name, args) -> Option<Result>`.
`call_builtin` chains them:

```rust
pub fn call_builtin(name, args) -> Result<DenseArray, RuntimeError> {
    if let Some(r) = math_builtins::try_call(name, args.clone()) { return r; }
    if let Some(r) = random_builtins::try_call(name, args.clone()) { return r; }
    if let Some(r) = ensemble_builtins::try_call(name, args.clone()) { return r; }
    // ... each module owns its responsibility; call_builtin stays narrow.
}
```

Adding a new family adds one line here + one new
module. The chain is open for extension, closed for
modification.

### 9. Bridge (same surface, different backend)

**When:** one abstraction (an op by name) has multiple
implementations (CPU, MLX).

**Example:** `dispatched_call(env, op_name, args)`
checks the active device and routes to the MLX backend
or the CPU `call_builtin`. Callers never branch on
device; the bridge does.

### 10. Split tests from work

**When:** a test function body is >50 LOC. Tests deserve
the same readability budget as production code.

**Example (Saga 16 step 002 -- `tsne_preserves_three_cluster_structure`):**
the cluster-fixture builder, the centroid helper, and
the distance helper are all separate small functions.
The test body is ~20 LOC of clear assertions.

## Anti-patterns

Things that signal "a pattern above would have been the
right move":

- **`#[allow(clippy::too_many_arguments)]`** -- always a
  smell. You're suppressing a lint that exists
  specifically to flag missing structs.
- **Complex tuple return type (4+ fields)** with a
  `type` alias to hide it -- the alias doesn't help the
  caller; name the fields in a struct.
- **Single function handling validation + allocation +
  initialization + work** -- phases unpacked, each one
  a separate function.
- **Mutable shared state threaded as `env.` across many
  functions** -- consider whether a scoped `Ctx` struct
  passed explicitly gives clearer dataflow.
- **Growing match arm with a growing body** -- extract
  per-arm helpers.
- **A file growing toward 500 LOC with a vague name** --
  ask what the first-class concept is; extract it to a
  named module.

## When to accept an existing FAIL

Pre-existing FAILs in this codebase (`model_dispatch.rs`,
`builtin_softmax`, `builtin_blobs`, etc.) are
historical. Don't silently make them worse, but don't
drop unrelated refactors into the middle of a feature
commit either. Two rules:

1. **New code must not introduce new FAILs.** If my
   commit ticks a counter upward, I fix it in the same
   commit. That's the baseline Saga 15 and 16 held.
2. **Touching a pre-existing over-budget function
   triggers a small extraction.** If I'm editing
   `apply_model` (already at 71 LOC) and adding 5 more
   lines, the right move is to extract the arm I'm
   editing into a helper. Don't grow the carrying file.

## Related

- `sw-checklist` -- the tool that enforces the budgets.
- `CLAUDE.md` -- project-level instructions that
  reference this doc.
- Memory: `feedback_sw_checklist_patterns.md` -- the
  durable preference note.
