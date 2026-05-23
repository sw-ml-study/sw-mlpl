# Loose Coupling: techniques for compact, readable code

This is the canonical reference for HOW to write code that
fits the project's [code metrics](code_metrics.md) gates
without playing tricks. The metrics (25 LOC per function,
7 functions per module, 7 modules per crate, 500 lines per
file) assume idiomatic formatting. When a function or
module is over budget, the answer is **never** to compress
whitespace or fight `rustfmt`. The answer is to restructure
the code so that each piece does one thing and the pieces
compose cleanly.

Two complementary lenses drive every refactor:

1. **Phase separation.** Code runs at one of four phases.
   Identify which phase each piece belongs to and move it
   there.
2. **Compose, don't compress.** A function is wide when it
   is doing two or three things. Split by responsibility,
   then compose the small pieces with delegation or iterator
   pipelines.

The rest of this document expands each lens into concrete
patterns and anti-patterns. The patterns are not optional --
they are the rules the metrics budget assumes.

## Phase 1: compile time

What can be computed when `cargo build` runs has zero
runtime cost. Anything constant -- lookup tables, default
configs, parser keyword sets, builtin dispatch tables --
belongs here.

**Patterns:**

- `const` items for primitive constants and `&[T]` slices of
  `Copy` data.
- `static` items for non-Copy data (e.g. owned `String`s, but
  prefer `&'static str` where possible).
- `const fn` helpers so derived constants stay
  compile-time-evaluable.
- `macro_rules!` and proc macros to generate repetitive
  boilerplate (impl blocks, builtin registrations).
- Generic functions with monomorphization: the same source
  specializes per concrete type, no runtime dispatch.
- Type-state encodings: encode invariants in the type system
  so the compiler enforces them at zero runtime cost.

**Example in this repo.** `mlpl-eval-core::inspect_groups`
exposes `BUILTIN_GROUPS: &[FnGroup]` as a `const`. The
inspection module just iterates over it -- the table itself
is compiled into the binary, no allocation, no setup.

**Anti-pattern:** building a lookup table inside the
function that uses it. Hoist to `const`.

## Phase 2: start-up

What happens once per process belongs here: registry
construction, parsing argv, loading config files, opening
log sinks. The cost is paid once at boot, amortized over
the lifetime of the process.

**Patterns:**

- `OnceLock<T>::get_or_init(closure)` for thread-safe
  lazy initialization (replaces `lazy_static` in stable
  Rust). Pay the closure cost on first access; subsequent
  accesses are a single atomic load.
- Builder pattern: `Foo::builder().with_x(...).build()` so
  construction is explicit and orderly. The builder is the
  start-up code; the resulting `Foo` is what runtime code
  uses.
- Init function called from `main` (or a top-level
  framework entry point) that wires together the
  long-lived state once.
- `Arc<T>` or `Rc<T>` to share start-up state with worker
  closures without rebuilding it.

**Example in this repo.** `mlpl-runtime::call_builtin`
dispatches by walking per-module `NAMES` constants. Those
constants are compile-time. The dispatch itself is a
straight-line match -- no per-call initialization.

**Anti-pattern:** repeating start-up work inside a hot
loop. If a function builds a `HashMap` on every call, hoist
to a `OnceLock` or pass the map in as an argument from the
caller.

## Phase 3: conditional

Branches taken based on a flag or feature -- selecting an
implementation, not transforming data. The branch happens
once per call, not per item.

**Patterns:**

- `#[cfg(feature = "X")]` and `#[cfg(target_arch = "...")]`
  for build-time conditionals. The compiler eliminates the
  unselected code; no runtime cost. Use this for: WASM vs
  native, optional integrations (`image-io`, `mlx`),
  test-only helpers.
- `Box<dyn Trait>` or generic parameters for runtime
  polymorphism. The dispatch table is set up at start-up;
  per-call cost is one indirect jump.
- A small `enum` + `match` for a closed set of choices. The
  compiler can inline arms; cost is one branch.
- Trait dispatch via methods on a configured struct: the
  struct is set up at start-up, the trait method is the
  per-call branch.

**Example in this repo.** `device("mlx") { body }` selects
the MLX runtime backend at start-up; inside the block, no
runtime "am I in MLX mode" check exists -- the dispatch
function pointer was set up when the block was entered.

**Anti-pattern:** an `if config.feature_x { ... } else { ...
}` check that runs on every iteration of a hot loop. Hoist
the choice ABOVE the loop and dispatch once.

## Phase 4: dataflow pipeline

Per-item transformation: stream of inputs in, stream of
outputs out. Each item flows through a sequence of stages.
This is the largest source of "wide function" mistakes
because programmers write nested loops instead of pipelines.

**Patterns:**

- Iterator combinator chains:
  ```rust
  input
      .iter()
      .filter(is_valid)
      .map(parse_entry)
      .map(normalize)
      .collect::<Vec<_>>()
  ```
  Each closure is a stage. Each stage is a small named
  function. The chain reads top-to-bottom like a sentence.
- `Result::and_then` and `Option::map_or_else` for
  fallible-stage chains. Each stage produces a `Result` or
  `Option`; the chain composes them without intermediate
  match statements.
- `fold` for stateful reductions (sum, max, accumulator).
- `flat_map` for nested iteration without nested `for`.
- `scan` for stateful per-item transforms.
- `try_fold` for early-exit reductions.

**Key rule: define stages separately from composition.**
The outer function is the *recipe* (which stages, in what
order). The stages themselves are *ingredients* (small
named helpers). When you read the outer function, you read
the recipe; you do not need to inline the ingredient
definitions.

**Example.** Refactoring a 27-line `decode_directory_to_record`:

```rust
// Bad: 27 lines in one function, four jobs interleaved.
fn decode_directory_to_record(dir: &Path, h: usize, w: usize) -> ... {
    // walk dir, filter images, decode each, build X/Y/names
    let mut paths = ...;
    for entry in fs::read_dir(dir)? { ... }
    paths.sort();
    if paths.is_empty() { return Err(...); }
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    let mut names = Vec::new();
    for path in &paths {
        x_data.extend_from_slice(&decode_and_resize(path, h, w)?);
        let name = ...;
        y_data.push(label_for(name) as f64);
        names.push(name.to_string());
    }
    build_record(...)
}
```

```rust
// Good: orchestrator reads as recipe, stages are leaves.
fn decode_directory_to_record(dir: &Path, h: usize, w: usize) -> ... {
    let paths = sorted_image_paths(dir)?;
    if paths.is_empty() { return Err(empty_dir_err(dir)); }
    let (x_data, y_data, names) = decode_paths(&paths, h, w)?;
    build_record(paths.len(), 3, h, w, x_data, y_data, names)
}

fn sorted_image_paths(dir: &Path) -> Result<Vec<PathBuf>, EvalError> { ... }
fn decode_paths(paths: &[PathBuf], h: usize, w: usize)
    -> Result<(Vec<f64>, Vec<f64>, Vec<String>), EvalError>
{
    paths
        .iter()
        .map(|p| decode_one(p, h, w))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip3()  // or fold into three Vecs
}
fn decode_one(path: &Path, h: usize, w: usize)
    -> Result<([f64; ...], f64, String), EvalError> { ... }
```

The orchestrator is 5 lines. Each helper is its own
short, named operation. Errors propagate via `?`. The
data flow is left-to-right through the iterator chain.

**Anti-pattern:** stateful loops with `let mut`
accumulators when an iterator combinator would do the
same job in one line.

## Building chains vs dispatching chains

A function that **builds** a value (an `Environment`, a
`ModelSpec`, a parser AST) is a *constructor* or *builder*.
A function that **dispatches** a value to one of N
implementations (the eval engine, the runtime registry,
the model-apply path) is a *router*. These are two
different concerns. **Do not put both in the same
function**, and prefer not to put both in the same module.

**Why this matters.** A "build + dispatch" function is
roughly twice as wide as either half on its own. It also
has twice as many reasons to change: a new builder pattern
forces a rewrite even though the dispatch is unchanged,
and vice versa.

**Signs you're mixing them:** the function name has
"build" or "construct" in the docs but the body has a
`match` over node kinds. Or the body starts with field
allocation and ends with a registry lookup. Split.

**Example in this repo.** Saga 32 step 002 split
`mlpl-runtime` into:

- `mlpl-runtime` (the dispatcher: `call_builtin` walks
  per-module `NAMES` constants and routes to the matching
  `try_call`).
- `mlpl-runtime-{math, data, dim-reduction}` (the
  per-family builtin bodies -- each one builds and returns
  a result).

The dispatcher reads as one function; each builder family
has its own crate with its own module organization. The
parts evolve independently.

## One responsibility per function

If you cannot name what a function does in five words
without "and", split it. The split usually clarifies the
OG function's name too.

**Test:**

- "parses input and validates and computes" -> three
  functions: `parse_input`, `validate_input`,
  `compute_result`.
- "loads dataset and builds record and writes to disk" ->
  three functions plus an orchestrator that calls them in
  order.

**Counter-test:**

- "evaluates an expression to a value" -- one
  responsibility, even if 24 lines.
- "renders a Lesson struct to HTML" -- one
  responsibility, even if it uses three different yew
  components inline.

If the function name needs an "and", the split is real.

## Strict dependency DAG

The crate-level and module-level dependency graphs must be
**acyclic**. A cycle is structural debt: it forces
unrelated changes to ship together, blocks crate splits,
and forces the build to recompile both sides of the cycle
together every time either side changes.

When a refactor reveals a cycle, **break the cycle**, do
not defer it. Two techniques:

1. **Trait-callback inversion.** Define a trait in the
   lower crate that the higher crate implements. The lower
   crate takes `&mut dyn Trait` and calls back without
   importing the upper crate's concrete types.
2. **Shim split.** Each cyclic function splits in two: a
   pure half (no callback) in the lower crate, and a
   wrapper half (resolves args, calls the pure half) in
   the higher crate.

See [feedback_no_cyclic_deps.md] in the agent memory for
the specific saga-32 cycle-breaking story.

## Anti-patterns (do NOT do these)

- `#[rustfmt::skip]` to keep wide struct literals on one
  line. The formatter is on the side of the next reader.
  If a struct literal is too wide for one line, the struct
  has too many fields. Split it.
- `#[allow(clippy::too_many_arguments)]` to dodge the
  argument-count lint. Bundle related args into a struct
  (see `LinearLoraInputs`, `AttentionInputs` in this repo).
- Workspace `rustfmt.toml` overrides that widen the
  per-line budget. That globalizes a per-function hack.
- Compressing `let x = expr1; let y = expr2;` chains into
  one line. Each `let` is a thought; compressing thoughts
  hides them. The line count goes down, the comprehension
  cost goes up.
- Adding `cfg!()` runtime checks where a `#[cfg(...)]`
  build-time gate would do (the runtime check pulls the
  dead branch into the binary).
- Hand-rolled loops with `let mut` accumulators when an
  iterator chain would express the same intent.
- Inline tests that push a production module over the
  function-count budget. Move tests to a `tests/` sibling
  file or split the inline `mod tests { ... }` across the
  appropriate phase boundary.

## When stuck: look for the phase mismatch

A function over budget that resists splitting usually has a
**phase mismatch**: it is doing per-call work that belongs
at start-up, or per-item work that belongs at compile time,
or vice versa. Walk the four phases (compile time, start-up,
conditional, dataflow) and ask which phase each piece of
the function's work actually belongs in. Move each piece
to its phase. The function will shrink, often dramatically.

A pure dataflow function over budget probably needs to be
split into multiple stages, each its own helper. A pure
start-up function over budget probably needs a builder
pattern. A mixed-phase function needs to be split by phase
first; the per-phase pieces are usually small enough.

## See also

- [code_metrics.md](code_metrics.md) -- the budgets and
  refactor triggers this document supports.
- agent memory entries `feedback_compose_dont_compress`,
  `feedback_phase_separation`, `feedback_no_cyclic_deps`,
  `feedback_sw_checklist_ratchet_down` -- the specific
  rulings that drove this document.
- [saga-tech-debt-paydown.md](saga-tech-debt-paydown.md) --
  the saga 32 plan, including its honest gap with the
  "halve both" target (the strategies in this doc are the
  carry-over for the follow-up saga).
