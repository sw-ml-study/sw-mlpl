# Saga: compiler-process-semantics

Lower the PROCESS entry/status/output builtins in the compile-to-Rust
path (`mlpl-lower-rs` -> `mlpl_rt`), matching interpreter semantics, so
a standalone compiled MLPL program is a well-behaved CLI: it can print
to stdout/stderr, read stdin, exit with a status code, and emit binary
stdout WITHOUT a spurious trailing text line. This is the last compiler
rung before a standalone compiled hexdump / WAV CLI (the
demo-file-processing capstone) becomes expressible; control flow
(`compiler-control-flow`) is the remaining sibling rung.

Current state (compiler-byte-io + compiler-text-conversions shipped):
bytes/bit/text conversions + `write_stdout` lower. But `print`,
`eprint`, `exit`, and `read_stdin` do NOT lower, and the generated
`main` ALWAYS does `println!("{}", result)` (mlpl-build project.rs
write_main_rs) -- so a program ending in `write_stdout(bytes)` prints a
spurious `ok(N)` text line after the binary output.

Interpreter reference (parity, must match): `eval_intercepts.rs`
(print/eprint/read_stdin/exit dispatch) + `eval_script.rs`
(read_stdin_to_string, eval_exit, eval_print). Output primitives RETURN
their argument unchanged (so they compose: `x = print(expr)` binds and
shows). Compiler dispatch to extend: `mlpl-lower-rs/src/fncall.rs`
(REGISTRY + Emit). Runtime home: a process module in `mlpl-rt-value`
(or `mlpl-rt-fsio`). Generated `main`: `mlpl-build/src/project.rs`.

Load-bearing parity rules:
- `print`/`eprint` write the value's Display to stdout/stderr and
  RETURN the value (composable), matching the interpreter's
  return-the-arg output primitives.
- `exit(code)` ends the process with that integer status; a compiled
  binary's exit code must equal the interpreter's.
- `read_stdin()` reads all of stdin to a `CVal::Str` (matching
  `read_stdin_to_string`), with the same empty/EOF behavior.
- The generated `main` must not append a spurious Display line after a
  process-effect program: binary stdout stays byte-identical.

Each step is TDD (RED failing e2e -> GREEN lowering -> refactor) with a
gated `MLPL_BUILD_TESTS=1` compiled-binary test. Hold or lower
sw-checklist each step.

## Steps

1. print-eprint -- lower `print/1` and `eprint/1`: runtime fns that
   write the `CVal` Display to stdout / stderr and return the value
   unchanged (composable), plus the `fncall` lowering. TDD + gated
   e2e: a compiled program prints a value to stdout and to stderr and
   the binding `x = print(v)` still yields `v`.

2. exit -- lower `exit/1` (and `exit/0` if the interpreter allows it):
   end the process with the integer status. TDD + gated e2e: a
   compiled program `exit(3)` returns exit code 3; a normal program
   returns 0. Match the interpreter's code semantics + argument
   validation.

3. read-stdin -- lower `read_stdin/0`: read all of stdin into a
   `CVal::Str`, matching `read_stdin_to_string` (empty on immediate
   EOF). TDD + gated e2e: a compiled program echoes piped stdin (e.g.
   `disp(read_stdin())`), byte-for-byte.

4. pristine-stdout -- fix the generated `main` (project.rs
   write_main_rs) so a process-effect program does NOT get a spurious
   trailing Display line: `write_stdout(bytes)` output is
   byte-identical with no `ok(N)` tail, while a value-returning
   program still shows its result (interpreter script parity). Decide
   the rule from the interpreter's `-f script` output behavior and
   pin it. TDD + gated e2e: byte-exact stdout for a binary-emitting
   program; unchanged text for a value program.

5. docs-close -- user + contract docs (lang-reference / the compiler
   capability doc, WHAT/HOW only), mark the compiler-process-semantics
   rung SHIPPED in docs/future-sagas-queue.md (compiler-control-flow
   the remaining rung to the file-processing capstone), refresh
   docs/companion-demo-file-processing.md (process semantics cleared),
   and the wiki errata if a compiled-capability claim flips. Hold
   sw-checklist. `--done`.
