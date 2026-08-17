# Saga: compiler-file-processing-builtins

Lower the ~12 builtins the ../demo-file-processing compiled wc / grep /
du tools need but the compile-to-Rust path does not yet support, in
wc -> grep -> du priority order. Traced from the wc/grep/du include
closures; each stops `mlpl-build` at the first unlowered builtin
(`take/3` is the current wall). Everything else those tools use
(`gt`/`lt`/`eq`, `rank`, `tally`, `read_bytes`, `file_size`, `ok`/`err`/
`?`, records, `if`/`while`, user fns) already lowers.

Each builtin is lowered to EXACT interpreter parity, with a gated
`MLPL_BUILD_TESTS=1` compiled-binary e2e. Add each to the fncall
REGISTRY (new Emit shape only if genuinely new) and the
dispatch_coverage_tests Builtin enum. Hold or lower sw-checklist per
step; after each push, refresh target/{release,debug} mlpl-repl AND
mlpl-build (keep-release-debug-binaries-fresh).

The missing set by tool:
- **wc**: `take` (bounded slice), `floor` (math), `type_of` (value kind
  -> string), `equal` (structural equality -> 0/1).
- **grep** adds: `str_concat`, `str_find`, `str_len`, `str_slice`,
  `str_split`.
- **du** adds: `fs_walk` (dir -> StrList), `list_get`, `list_len`,
  `concat` (array concat).

RISK / dependency to confirm from the interpreter map: the string /
list / fs builtins may need a `CVal` StrList variant if `mlpl-rt-value`
`CVal` only has Str/Arr/Record/Result today. If so, the grep/du steps
add that variant first. `type_of`/`equal`/`str_*`/`list_*`/`fs_walk`
are eval-layer (Value) in the interpreter; the compiler must reproduce
their semantics on `CVal` (pure, no interpreter/parser).

## Steps

1. wc-take-floor -- lower `take` (confirm 2-arg and/or 3-arg slice
   semantics from the interpreter; the downstream uses `take/3` =
   `take(a, offset, length)`) and `floor/1` (elementwise math). Both
   pure DenseArray/math (reuse a runtime primitive or mirror it in
   mlpl-rt). TDD + gated e2e.

2. wc-type-equal -- lower `type_of/1` (value kind -> a `CVal::Str`,
   exact interpreter strings) and `equal/2` (structural equality of two
   values -> scalar 0/1, mlpl-value-structural parity). Operate on
   `CVal`. TDD + gated e2e. After this, wc.mlpl should compile.

3. grep-string-ops -- lower `str_concat`, `str_find`, `str_len`,
   `str_slice`, `str_split` on `CVal::Str` (exact interpreter
   semantics + return types: Str / scalar / StrList / Result). Add a
   `CVal` StrList variant if needed for `str_split`. TDD + gated e2e.
   After this, grep.mlpl should compile.

4. du-list-fs -- lower `list_get/2` + `list_len/1` (StrList ops) and
   `fs_walk` (sandboxed dir -> StrList, via the compiled fs sandbox
   root) and `concat` (array concat). TDD + gated e2e. After this,
   du.mlpl should compile.

5. docs-close -- document the newly compile-capable builtins
   (lang-reference / compiler capability doc, WHAT/HOW only), mark the
   saga SHIPPED in docs/future-sagas-queue.md (the remaining GNU-clone
   gates are the dedicated CLI entry points + arg-driven paths /
   top-level unwrap / chunked-stdin streaming), refresh
   docs/companion-demo-file-processing.md, update the wiki errata if a
   compiled-capability claim flips, and queue the fncall.rs
   handler/dispatch split tech-debt. `--done`.
