# sw-checklist Mitigations

Specific prevention and fix strategies for each warning
and failure type. Ordered by frequency (most common first).

Current counts (2026-05-25): 129 FAIL, 463 WARN.

## Function LOC (340 WARN, 24 FAIL)

**Warning >25 lines, FAIL >50 lines. Target <=20.**

The single largest category. Most over-budget functions
are doing two or more of these in one body:

| Symptom | Fix |
|---------|-----|
| Validation + logic | Extract a `validate_*` function that returns a typed result; the main fn calls it and proceeds |
| Build + use | Extract the builder (struct-literal construction, callback wiring) into a named builder fn; the caller uses the result |
| Match arms >3 lines each | Each arm becomes a named helper; the match becomes a dispatcher |
| Inline HTML template >15 lines | Extract sub-templates as sibling functions or child components |
| Error construction >2 lines | Extract `fn make_err(...)` helpers; the call site is one line |
| `let x = ...; Struct { x }` | Use `Struct { field: expr() }` directly -- saves one line per field |
| Callback wrapping `\|_: T\| cb.emit(())` | Accept `Callback<T>` in the prop type instead of `Callback<()>`; eliminates the wrapper |

**Prevention:** Before writing a function, list the steps
it will perform. If >4 steps, split into two functions
before writing any code.

## Module Function Count (89 WARN, 60 FAIL)

**Warning >4 fns, FAIL >7 fns. Target <=4.**

| Symptom | Fix |
|---------|-----|
| `#[cfg(test)] mod tests` inflates count | Move tests to a separate test file (`tests/` dir or `*_tests.rs` sibling) |
| Cluster of related helpers | Group into a sub-module with its own file; re-export from parent |
| Read + write + predicate per concept | Reduce API surface: remove trivially-inlineable predicates; merge read+write if they always pair |
| N callback builders | Use an enum + single dispatcher instead of N separate functions |
| Component + props + N render helpers | Keep component + 1 helper max per file; extract others to siblings |

**Prevention:** Count functions at design time. A module
gets the component fn + at most 3 helpers = 4 total.
Tests go elsewhere. If the design needs >4, split into
two modules by sub-concern before writing code.

## File LOC (16 WARN, 23 FAIL)

**Warning >350 lines, FAIL >500 lines. Target <=250.**

| Symptom | Fix |
|---------|-----|
| Inline string content (demo intros, help text) | Move to text files; use `include_str!` or `build.rs` codegen |
| Large `const` arrays (DEMOS, PATHS) | Split by topic into sibling files; facade re-exports the combined array |
| Fat `impl` blocks | Split impl across files (Rust allows multiple `impl Foo` blocks in the same crate) |
| Mixed concerns in one file | Each concern (validation, dispatch, rendering) gets its own file |

**Prevention:** Estimate LOC before writing. If content
strings will exceed 100 lines, externalize to a text file.
If logic will exceed 200 lines, plan two modules.

## Crate Module Count (17 WARN, 9 FAIL)

**Warning >4 modules, FAIL >7 modules. Target <=4.**

| Symptom | Fix |
|---------|-----|
| Flat `src/*.rs` layout with many files | Group into sub-directories; each directory = 1 module from the crate's perspective |
| Every concern gets its own top-level module | Cluster related concerns into one directory-module with internal sub-modules |
| Facade + N sub-modules all at crate root | The facade IS the crate root (`lib.rs`); sub-modules live in directories |

**Prevention:** Before adding a module to a crate, check
`lib.rs` module count. If at 4, create a sub-directory
for the concept cluster instead of a new top-level module.

## Clippy Allows (12 FAIL)

**Every `#[allow(clippy::*)]` is a FAIL. Target: zero.**

| Symptom | Fix |
|---------|-----|
| `#[allow(clippy::too_many_arguments)]` | Bundle args into a struct (3-5 fields each); pass the struct |
| `#[allow(clippy::type_complexity)]` | Introduce a type alias for the complex type |
| `#[allow(clippy::needless_borrow)]` | Remove the unnecessary `&`; clippy is usually right |
| Any `#[allow(...)]` to silence a warning | Fix the underlying issue; the allow hides a real problem |

**Prevention:** Never write `#[allow(clippy::*)]`. If
clippy complains, the code has a design problem. Fix the
design, not the lint.

## Rust Edition (1 FAIL)

Check `Cargo.toml` for `edition = "2021"` or later.

## Copyright (1 WARN)

Add a copyright notice to the crate root or `Cargo.toml`.

## General principles

1. **Warnings and FAILs are both problems** -- different
   severity, same obligation to fix. Fix warnings first;
   FAILs shrink naturally.

2. **Every commit must reduce the total problem count.**
   If a feature adds warnings, the design is wrong --
   redesign before committing.

3. **Design to the target, not the limit.** A module at 4
   fns can absorb one addition. A module at 7 cannot.

4. **Count before you code.** Functions, lines, modules --
   know the numbers before writing. If the design won't
   fit, split before you start.

5. **Content is not code.** String literals, static arrays,
   help text, demo descriptions belong in text files or
   generated constants. Inline content inflates every
   metric.
