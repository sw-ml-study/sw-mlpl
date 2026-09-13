# Saga: axis-naming-unification (resumed)

Resume of the paused axis-naming-unification saga. Original steps 1-4 shipped
(see docs/unifying-plan.md and .agentrail-archive/axis-naming-unification-*):
`label` / `reduce` / `reshape_labeled` now accept a bracketed name list, a
comma-string, or a computed value interchangeably via the shared
`mlpl_axes::AxisSpec` / `axis_adapter::axis_names_of` -- the label-vs-reduce
inconsistency is resolved in the interpreter. This resumes the remaining
polish, compiler parity, surface sweep, and close.

## Steps

1. errors-and-docs -- unify the ad-hoc axis errors behind one
   `AxisError`-derived message naming all accepted forms; document the
   canonical bracketed-name form in docs/lang-reference.md + docs/glossary.md
   (WHAT/HOW only) and the wiki errata. TDD where messages/behavior are pinned.

2. compiler-parity -- lower a named `reduce_add(x, names)` resolving names ->
   indices at lower time via `Ctx.known_labels` (LowerError when not static);
   extend label lowering (`extract_label_list`) to accept a constant-foldable
   `StrList`. Register in `REGISTRY`, add the `dispatch_coverage_tests` Builtin
   variant, update `CVAL_BUILTINS` if a new CVal shape appears, gated
   `MLPL_BUILD_TESTS=1` e2e. See docs/unifying-plan.md section 5.

3. surface-sweep -- route `compress`, `drop`, `reduce_add` and any other
   axis-selecting builtin through the shared `AxisSpec` path; enumerate from
   the eval dispatch table + `supported_builtin_names()` and `log` any left out.

4. downstream-relay-and-close -- confirm `../demo-ml-utils`
   `probes/named-axis-reduce.mlpl` flips green with the shipped reduce-StrList;
   relay the demo edits (docs/unifying-plan.md section 10); refresh CHANGES;
   update docs/glossary + wiki; mark the resumed saga shipped in
   docs/future-sagas-queue.md. `--done`.
