# Saga: axis-naming-unification

Make `label`, `reduce`, `reshape_labeled`, and the other axis-selecting
builtins accept the same axis-selector forms (bracketed name list,
comma-string, integer indices) across the interpreter and the
compile-to-Rust path, via one shared `AxisSpec` / `AxisNames` type and a
single resolver. Full design, type map, phase rules, non-breaking impact
analysis, and downstream demo guidance: `docs/unifying-plan.md`.

Backward-compatible throughout: every form valid today stays valid; the
comma-string is kept as accepted sugar. Stage 1 alone fixes the
`label`-vs-`reduce` inconsistency the CNN blog post surfaced. TDD
(Red/Green/Refactor) on every step.

## Steps

1. axisspec-core -- Add the shared pure `AxisSpec` (Names | Indices) and
   `AxisNames` types plus one `resolve(&DenseArray) -> Vec<usize>` to
   `mlpl-array` (or a small focused sibling module `axis_spec.rs`),
   splitting parse / validate / resolve to stay under the metric gates.
   RED first: unit tests for missing name, no-labels, out-of-rank,
   duplicate, empty. No builtin wired yet.

2. interpreter-reduce-strlist -- Add the eval-side `axis_spec_of(&Value)`
   adapter (exhaustive match: StrList -> Names, Str -> Names via
   comma-split sugar, Array -> Indices) and make `reduce` / `reduce_add`
   resolve through it, so `reduce(:add, x, ["a","b"])` works. Keep the
   comma-string and integer-index forms. TDD.

3. interpreter-label-evaluate -- Add `axis_names_of(&Value)` and make
   `label` / `relabel` / `reshape_labeled` EVALUATE their name argument
   and accept a `StrList` or comma-`Str`, preserving identical behavior
   for the literal case. Add the table-driven PARITY test asserting the
   axis builtins accept the same set of forms. TDD.

4. errors-and-docs -- Replace the ad-hoc per-builtin errors with one
   `AxisError`-derived message that names all accepted forms (retire
   "expected an array value, got a string"). Update
   `docs/lang-reference.md` + `docs/glossary.md` (WHAT/HOW only) and the
   wiki errata to document the canonical bracketed-name form.

5. compiler-parity -- Lower a named `reduce_add(x, names)` by resolving
   names -> indices at lower time via `Ctx.known_labels` (LowerError when
   not statically known); extend label lowering (`extract_label_list`) to
   accept a constant-foldable `StrList`. Register in `REGISTRY`, add the
   `dispatch_coverage_tests` Builtin variant, update `CVAL_BUILTINS` if a
   new CVal shape appears, and add the gated `MLPL_BUILD_TESTS=1` e2e.

6. surface-sweep -- Route `compress`, `drop`, and any other
   axis-selecting builtin through the shared `AxisSpec` path; enumerate
   candidates from the eval dispatch table + `supported_builtin_names()`
   and `log` any deliberately left out.

7. downstream-relay-and-close -- Confirm `../demo-ml-utils`
   `probes/named-axis-reduce.mlpl` flips green; relay the section-10 demo
   edits (demos/cnn/06 named axes, src/cnn conv_layer unification, the
   computed-name-vector demo). Refresh CHANGES.md, mark the saga shipped
   in `docs/future-sagas-queue.md`, update `docs/saga.md`. `--done`.
