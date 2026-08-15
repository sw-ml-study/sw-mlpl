# Example runner scripts

Small wrappers for listing, compiling, and running the `.mlpl`
programs under `examples/`. They resolve the repo layout from their own
location, so you can call them from any directory, and they build the
release tools (`mlpl-repl`, `mlpl-build`) on first use.

## Scripts

| Script | What it does |
| --- | --- |
| `list.sh` | List every `examples/**/*.mlpl` with its line count. |
| `run.sh [--interp\|--native] <file.mlpl>` | Run one program. |
| `compile.sh <file.mlpl> \| --all` | Compile to a native binary in `bin/`. |

```bash
examples/scripts/list.sh
examples/scripts/run.sh examples/compile-cli/hello.mlpl            # interpret
examples/scripts/run.sh --native examples/compile-cli/hello.mlpl  # compile + run
examples/scripts/compile.sh --all
```

## Two ways to run

- **Interpreted** (default, `run.sh` or `run.sh --interp`) uses
  `mlpl-repl -f`. It supports the whole language.
- **Native** (`run.sh --native`, or `compile.sh`) uses `mlpl-build` to
  produce a self-contained executable, then runs it. `mlpl-build`
  compiles for the HOST target, so you get a Mach-O binary on macOS and
  an ELF binary on Linux automatically -- no flags, no cross-compile.
  The compiled path covers a subset of the language (no
  `repeat` / `train` / `for` / `grad` / Model DSL, and a compiled
  function may not read globals). Programs that do not lower are
  reported and skipped by `compile.sh --all`; run those interpreted.

## The `bin/` directory

Compiled output lands in `examples/bin/`, which is gitignored
(`examples/bin/.gitignore` keeps the directory but ignores its
contents). Binaries are host-specific and rebuilt locally, so they are
never committed -- a macOS checkout holds arm64 Mach-O binaries, a Linux
checkout holds x86_64 ELF binaries, from the same sources.

## Running the WebAssembly output

`mlpl-build --target wasm32-unknown-unknown -o prog.wasm` (shown in
`examples/compile-cli/README.md`) produces a **browser/embedding
module**, not a command-line program: the `wasm32-unknown-unknown`
target has no WASI, so there is no `_start` and no stdout -- a CLI
runtime like `wasmtime prog.wasm` cannot run it. It is meant to be
loaded by a JavaScript host (the same way the web playground loads its
wasm bundle).

To run compiled MLPL as a standalone command-line wasm program you need
a WASI target and a WASI runtime instead:

```bash
rustup target add wasm32-wasip1
mlpl-build examples/compile-cli/hello.mlpl --target wasm32-wasip1 -o /tmp/hello.wasm
wasmtime /tmp/hello.wasm
```

That path is not provisioned in this checkout by default (no WASI target
installed, no `wasmtime` on PATH). For everyday use prefer the native
binaries above; reach for wasm only when you specifically need a
portable module.
