# Saga: compiler-source-loading

demo-file-processing's compiled programs fail because `mlpl-build`
does not resolve `include` (the compiler rejects `Expr::Include`).
This is the earliest compiler-parity gate (also blocks
demo-extensions' compiler work). Make the compiler resolve
includes with the SAME semantics as the interpreter's script mode.

Approach (AST-level, matches the interpreter): the interpreter's
script mode resolves includes via `mlpl-source-loader::expand()`
over a filesystem `SourceProvider` (sandbox: canonicalized root
containment, absolute-path + escape rejection, load-once,
cycle-chain errors), producing flattened statement chunks. Wire the
SAME `expand()` into `mlpl-build`'s front-end, flatten the chunks
to `Vec<Expr>`, and lower them directly with
`lower_with_config(rt_path = ::mlpl::__rt)` (the path the `mlpl!`
macro already uses), emitting the lowered block into the temp
project's main.rs. This bypasses the text->macro path (which cannot
take resolved statements) and is the foundation the later rungs
(FnDef, control-flow, byte/bit IO) lower from too.

- New mlpl-build module (source_load.rs): a minimal FsProvider
  (replicated from mlpl-cli's, sandbox identical) + load_stmts(
  input, source_dir) -> Vec<Expr> via expand()+flatten.
- run(): read+expand -> lower once (validation + codegen) ->
  write_main_rs(lowered tokens). Add --source-dir (optional),
  mirroring script mode; default sandbox root is the input's dir.
- Keep template/Cargo.toml deps (temp project already deps the
  `mlpl` facade; lowered `::mlpl::__rt::...` resolves).

Non-web (native compiler), so NO pages deploy.

## Steps
1. include-expand -- source_load.rs (FsProvider + load_stmts),
   run()/write_main_rs switch to expand+lower, --source-dir arg;
   keep template_tests green; e2e build test with an include
   (MLPL_BUILD_TESTS=1) + a unit test on load_stmts.
2. close -- update companion-demo-file-processing.md + queue
   (compiler-source-loading SHIPPED; next rung compiler-functions),
   wiki/q-and-a if warranted; --done.
