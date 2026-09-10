# Unifying Plan: type-safe axis-naming semantics

Status: PROPOSED (planning doc; not yet implemented). This file is a
plan, so it may name stages, ordering, and future work -- it is not a
user-facing surface. Chronology and roadmap belong here, not in the
language reference.

## 1. The problem, precisely

Two builtins name axes, and they disagree on the accepted spelling. Each
rejects the other's form:

```
label(x, ["out_y", "out_x", "channel"])   # OK
label(x, "out_y,out_x,channel")           # ERROR: second argument must be
                                          #        a bracketed list of string literals

reduce(:add, x, "channel,kernel_y")       # OK
reduce(:add, x, ["channel", "kernel_y"])  # ERROR: expected an array value, got a string
```

The user-visible effect: a single showcase example needs both spellings
two lines apart, which makes the language look arbitrary. This is the
same shape as the earlier `reduce_add`-vs-`reduce` complaint, relocated to
`label`-vs-`reduce`.

### 1.1 Root cause is NOT "no string-array value"

A first-class string list already exists on both sides:

- Interpreter: `Value::StrList { items: Vec<String> }`
  (`components/eval-types/crates/mlpl-eval-types/src/value.rs:107`). A
  bracketed all-string literal `["a","b"]` already evaluates to a
  `StrList` (`components/eval/crates/mlpl-eval/src/eval.rs:69-77`).
- Compiler: `CVal::StrList(Vec<String>)`
  (`components/native-rt/crates/mlpl-rt-value/src/value.rs:21`).

The real causes are two independent asymmetries:

1. **Phase asymmetry (interpreter).** `label`/`relabel`/`reshape_labeled`
   inspect the raw AST of their name argument and require it to be an
   `Expr::ArrayLit` of `Expr::StrLit`; they never evaluate it
   (`components/eval/crates/mlpl-eval/src/fncall_axes.rs:100-105,130-141`).
   `reduce` does the opposite: it evaluates its axis argument to a
   `Value` and branches on the kind -- `Value::Str` -> comma-split label
   lookup, anything else -> `into_array()` positions, and `StrList` falls
   into the "anything else" arm and is rejected with `ExpectedArray`
   (`components/eval/crates/mlpl-eval/src/eval_reduce.rs:92-118`).
   So `label` cannot take a *value* (a variable holding `["a","b"]`), and
   `reduce` cannot take the *bracketed* form at all.

2. **Surface asymmetry (each builtin re-implements acceptance).** There
   is no shared notion of "a selection of axes". Each site hand-rolls its
   own parsing, so they drift.

3. **Interpreter/compiler asymmetry.** The generic `reduce` is
   interpreter-only and supports named axes; the compiler lowers only
   `reduce_add` with a *numeric* axis and has no named-axis reduce
   (`components/syntax-codegen/crates/mlpl-lower-rs/src/fncall.rs:191-195`).
   Meanwhile the compiler *does* track labels statically
   (`Ctx.known_labels`, `model.rs:86`) and does static matmul
   contraction-axis checking (`fncall.rs:402-421`).

### 1.2 The three surface forms in play

Any axis argument today is one of:

| Form                | Evaluates to        | Meaning        |
|---------------------|---------------------|----------------|
| `["out_y","out_x"]` | `Value::StrList`    | axes by name   |
| `"out_y,out_x"`     | `Value::Str`        | axes by name   |
| `[0, 1]`            | `Value::Array`      | axes by index  |

The goal is that every axis-naming builtin accepts all three where they
make sense, via one shared, exhaustively-typed path.

## 2. Design principles

Drawn from `docs/code_metrics.md` and `docs/loose-coupling.md`:

- **Define once, invoke many.** One type for "a set of axes" and one
  resolver function; every builtin calls it. They cannot drift because
  there is a single source of truth.
- **Parse, then validate, then resolve.** Separate `Value -> AxisSpec`
  (parse) from `AxisSpec + array -> Vec<usize>` (resolve). Pure
  functions, no effects.
- **Exhaustive enums.** Model the forms as a closed enum so the compiler
  forces every call site (interpreter and compiler) to handle every form.
- **Phase separation.** Keep compile-time label knowledge (static
  inference, static checks) distinct from run-time resolution. The
  compiler and interpreter share the *resolution semantics* but differ in
  *when* they run it.
- **Tests pin the requirement.** A parity test asserts that the axis
  builtins accept the *same* set of forms, so a future edit that
  re-introduces divergence fails.

## 3. The type-safe core

### 3.1 Two small, pure types (shared)

Placed in `mlpl-array` (the crate both the interpreter and the compiler
already depend on, and the one that owns `DenseArray.labels` and
`with_labels`, `components/array/crates/mlpl-array/src/dense.rs:18`,
`indexing.rs:40-49`). No dependency on `Value` or `CVal`, so both sides
can use it without a cycle.

```rust
/// A selection of axes to operate over (reduce, compress, drop, ...).
#[non_exhaustive]
pub enum AxisSpec {
    Names(Vec<String>),    // resolve against an array's labels
    Indices(Vec<usize>),   // positional
}

/// A list of axis names to ATTACH (label / relabel / reshape_labeled).
/// Inner None leaves that axis positional.
pub struct AxisNames(pub Vec<Option<String>>);
```

### 3.2 One resolver (shared)

The single place that turns names into positions, used by every
axis-selecting builtin on both sides:

```rust
impl AxisSpec {
    /// Resolve to concrete axis indices against a labeled array.
    pub fn resolve(&self, arr: &DenseArray) -> Result<Vec<usize>, AxisError> {
        match self {
            AxisSpec::Indices(ix) => validate_in_rank(ix, arr.shape().rank()),
            AxisSpec::Names(names) => {
                let labels = arr.labels()
                    .ok_or(AxisError::NamedAxisButNoLabels)?;
                names.iter()
                    .map(|n| labels.iter()
                        .position(|l| l.as_deref() == Some(n.as_str()))
                        .ok_or_else(|| AxisError::NoAxisNamed(n.clone())))
                    .collect()
            }
        }
    }
}
```

`AxisError` is a small typed error the interpreter and compiler each map
into their own error enum (`EvalError` / `LowerError`).

### 3.3 Thin per-side adapters (parse)

Each side has ONE function that turns its value into an `AxisSpec`. These
are the only places that mention `Value` / `CVal`, and each is an
exhaustive match so a new value variant is a compile error until handled.

Interpreter (`Value -> AxisSpec`):

```rust
fn axis_spec_of(v: &Value) -> Result<AxisSpec, EvalError> {
    match v {
        Value::StrList { items } => Ok(AxisSpec::Names(items.clone())),
        Value::Str(s)            => Ok(AxisSpec::Names(split_comma(s))),   // sugar
        Value::Array(a)          => Ok(AxisSpec::Indices(as_indices(a))),
        other => Err(EvalError::ExpectedAxisSpec(other.value_kind().into())),
    }
}
```

`AxisNames` has a sibling parser (`StrList` or comma-`Str`, no index
form). `label` becomes: `eval` the argument, then `axis_names_of(&value)`
-- so it accepts a computed value, not only a literal.

## 4. What each builtin does after unification

| Builtin              | Accepts (names)          | Accepts (indices) | Notes                          |
|----------------------|--------------------------|-------------------|--------------------------------|
| `label` / `relabel`  | `["a","b"]`, `"a,b"`     | n/a               | now EVALUATES its arg          |
| `reshape_labeled`    | `["a","b"]`, `"a,b"`     | n/a               | same parser as label           |
| `reduce` / `reduce_add` | `["a","b"]`, `"a,b"`  | `[0,1]`           | now accepts StrList            |
| `compress` / `drop`  | `["a"]`, `"a"`           | `[0]`             | fold into the shared path      |
| `transpose_axes`     | (indices today)          | `[1,0]`           | optionally accept names later  |

The comma-string stays valid everywhere as accepted sugar (no forced
migration; deprecation timing is the maintainer's call). The bracketed
name list becomes the canonical, documented form.

## 5. The interpreter/compiler phase split (the subtle part)

The interpreter is fully dynamic: it resolves names against
`arr.labels()` at run time. The compiler wants names *statically* so it
can keep `known_labels` inference and static matmul checks. Resolve this
explicitly rather than papering over it:

- **label lowering.** Today it requires an AST literal
  (`extract_label_list`, `fncall.rs:425-437`). Extend it to also accept a
  *constant-foldable* `StrList` expression (a literal, or an ident bound
  to a literal in `known_labels`). A genuinely dynamic name list ->
  `LowerError::LabelsMustBeStatic` with a clear message. The interpreter
  has no such restriction. This is a deliberate, documented phase
  difference, not a bug.
- **named reduce lowering.** Add a lowered named `reduce_add(x, names)`
  that resolves `names` to indices at LOWER time using `known_labels`
  (`labels_of`, `fncall.rs:441-459`). If the labels are not statically
  known -> `LowerError::AxisNamesNotStaticallyKnown`. A later option is a
  runtime resolver in `mlpl-rt` that reads `DenseArray.labels` for full
  dynamic parity; only add it if compiled code needs dynamic names.

Stated as an invariant: **the resolution SEMANTICS are shared (Section
3.2); only the PHASE differs** -- run time in the interpreter, lower time
(static) in the compiler.

## 6. Staging (each stage ships and is tested on its own)

1. **Interpreter unification (fixes the reported bug).** Add `AxisSpec` /
   `AxisNames` + `resolve` to `mlpl-array`; add `axis_spec_of` /
   `axis_names_of` to the eval layer; make `reduce` accept `StrList`;
   make `label`/`relabel`/`reshape_labeled` evaluate their argument and
   accept `StrList` or comma-`Str`. Fully backward-compatible. Add the
   parity test (Section 7). Retire the misleading "expected an array
   value, got a string" wording.

2. **Error and doc pass.** Replace ad-hoc errors with one
   `AxisError`-derived message that names all accepted forms. Update
   `docs/lang-reference.md` and `docs/glossary.md` to document the
   canonical bracketed-name form (WHAT/HOW only; no roadmap in the
   user-facing docs). Mark comma-string as accepted sugar.

3. **Compiler parity.** Add a lowered named `reduce_add` (static
   name->index via `known_labels`); extend label lowering to accept a
   constant-foldable `StrList`. Register in `REGISTRY`
   (`fncall.rs:100-128`), add the `Builtin` coverage variant
   (`tests/dispatch_coverage_tests.rs`), update `CVAL_BUILTINS` if a new
   CVal-returning shape appears (`cval_lower.rs:35-57`), add the gated
   `MLPL_BUILD_TESTS=1` e2e parity check.

4. **Surface sweep.** Route `compress`, `drop`, and any other
   axis-selecting builtin through the shared `AxisSpec` path so the whole
   surface is consistent. Enumerate them from `supported_builtin_names()`
   and the eval dispatch table; log any deliberately left out.

5. **Optional, larger, separate saga.** Broader `Str`/`StrList` sequence
   ergonomics (indexing, mapping, joining string lists) is a different
   concern from axis naming. Note it here; do not fold it in.

## 7. How we prove it (tests pin the requirement)

A table-driven parity test enumerates {builtin} x {form} and asserts
acceptance is identical where the form is meaningful:

```
forms   = [ names_list ["a","b"], comma_str "a,b", index_list [0,1] ]
builtins= [ reduce, reduce_add, label*, reshape_labeled* ]   (* = names only)
assert: every (builtin, applicable form) evaluates without a kind error,
        and produces the SAME axis set as the other forms.
```

Plus: unit tests on `AxisSpec::resolve` (missing name, no-labels,
out-of-rank, duplicates, empty), and a compiler e2e that a named
`reduce_add` compiles and matches the interpreter. The parity test is the
guard that a future edit cannot silently re-introduce divergence.

## 8. Risks and edge cases

- **Ambiguity.** `[0,1]` is indices, `["a","b"]` is names -- distinct
  `Value` variants, so no ambiguity; a numeric axis for `label` stays an
  error (labels are names, not positions).
- **Compiler static-vs-dynamic names.** The one real semantic difference;
  handled by an explicit `LowerError` (Section 5), not silent divergence.
- **Backward compatibility.** Every form valid today stays valid; only
  new forms are added. No program breaks.
- **`mlpl-array` budget.** The new types are a small focused module
  (`axis_spec.rs`), designed to the gate (<=4 fns/module, <=25 LOC/fn);
  the resolver splits parse/validate/resolve to stay under budget. Check
  the crate's module count before landing (Crate-Module-Count gate).
- **`StrList` printing / round-trip.** Unchanged; we only add consumers,
  not new producers.

## 9. Impact on existing docs, demos, and the blog (non-breaking)

The plan is purely additive: every form valid today stays valid, and the
comma-string is kept as accepted sugar (Section 10 non-goals). Verified
against the current content:

- Every axis form used by shipped demos, examples, and the CNN blog post
  is an EXISTING valid form -- `reduce(:add, x, "channel")`,
  `reduce(:add, x, [2,3,4])`, `label(windows(...), [...])`,
  `reduce(:add, w*p, "channel,kernel_y,kernel_x")`. All keep working
  unchanged.
- The currently-erroring bracketed `reduce(:add, x, ["a","b"])` form
  appears nowhere runnable -- only in this plan (as the ERROR example) and
  in `docs/research4.txt` (a review doc). No demo, example, or test uses
  it, so stage 1 flips no red/green expectation.
- No test pins the current rejection of the bracketed form, and no test
  pins the "expected an array value, got a string" wording, so stages 1-2
  do not break a test.
- Making `label` evaluate its argument is behavior-preserving for the
  literal case (`["a","b"]` evaluates to a `StrList`, which the new parser
  accepts and resolves identically); it only ADDS the ability to pass a
  computed value.

The one downstream follow-up is editorial, not a break: a blog/doc that
today notes "the two spellings differ" would become stale once the
unification ships. Such text is correct for the current release and only
needs a small update when stage 1 lands -- normal doc lifecycle.

## 10. Non-goals

- Not changing how labels are stored (`Option<Vec<Option<String>>>` on
  `DenseArray` stays).
- Not introducing labels onto `Shape` or onto `CVal`.
- Not a general string-sequence overhaul (Section 6, stage 5).
- Not forcing removal of the comma-string form.
