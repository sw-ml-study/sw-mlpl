# Compiler coverage: what compiles to a native binary

`mlpl-build` is a **compile-to-Rust** path: MLPL source is lowered to
generated Rust over a `CVal` value model (`Arr` / `Str` / `StrList` /
`Record` / `Result` / `Bytes`), then `cargo`-built into a standalone
native binary. It compiles a defined SUBSET of the language -- the
"array + bytes + file + stdin + process + control-flow" core that a
CLI needs -- and gives a precise `Unsupported` error for the rest.

This doc is the human-readable boundary. The machine-checked half is
`components/syntax-codegen/crates/mlpl-lower-rs/tests/coverage_boundary_tests.rs`,
which asserts a pinned interpreter-only set stays disjoint from the
compiler's `supported_builtin_names()`; and `dispatch_coverage_tests.rs`,
which asserts every registered builtin actually lowers.

## What compiles

- **Arithmetic + operators**: `+ - * /`, unary minus, and the infix
  comparisons `< > <= >= == !=` (0/1 masks), with length-1 broadcast.
- **Control flow**: `if` / `else`, `while`, `break`/`continue` inside a
  `while`, `?` (Result propagation).
- **User functions**: `def u:name(...)` and calls. (Namespaces beyond
  `u:` are interpreter-only for now -- see below.)
- **Values**: numbers, strings, records + field access, `ok`/`err`
  Results, `StrList` literals, `Bytes`.
- **Array ops**: `shape`, `rank`, `transpose`, `reshape`,
  `reshape_labeled`, `label`/`relabel`, `reduce_add` (+ axis),
  `matmul`, `iota`/`range`, `tally`, `take`, `at`, `floor`,
  `type_of`, `equal`.
- **Bit ops**: `band bor bxor bnot popcount shl shr bmask bits from_bits`.
- **String ops**: `str_len str_concat str_find str_slice str_split`,
  `tokenize_bytes`, `decode_bytes`, `to_int`, `disp`.
- **I/O**: `read_bytes` (whole + range), `write_bytes`, `append_bytes`,
  `file_size`; `read_stdin`, `read_stdin_chunk`; `print`, `eprint`,
  `exit`, `args`, `arg` -- with pristine stdout (`finish_program`).

The definitive list is the `REGISTRY` in `mlpl-lower-rs/src/fncall.rs`
(operators are lowered directly, not via the registry).

## What does NOT compile

### A. Not yet lowered, but sensible to add (mechanical gaps)

- **Namespaces beyond `u:`** -- `ns:name` defs/calls and `:ns:name`
  refs work in the interpreter but the compiler still special-cases
  `u:` (queued: namespace-compiled-lowering / Phase 1b).
- **`for` / `repeat`** row iteration (only `if`/`while` lower today).
- **`StrList` / `du` builtins** -- `list_len`, `list_get`, `fs_walk`,
  `concat` (queued: du-list-fs).
- **Remaining pure array / codec builtins** -- `gather_rows`,
  `compress`, `grade_up`/`grade_down`, `rotate`, `flatten`, `sort`,
  JSON/TOML codecs. Each is "port to `mlpl-rt` + one registry row".

### B. Interpreter-only by nature

- **ML / autograd / training** -- `grad`, `adam`, `momentum_sgd`,
  `train`, `experiment`, `chain`/`linear`/`apply`, `param`/`tensor`.
  A tape-based reverse-mode engine; a compiled form means shipping a
  compiled autograd runtime or codegen'ing backward passes -- a
  separate, large project, and rarely what you want (training runs
  interpreted / on GPU).
- **Visualization** -- `svg`, `dataflow`, and the analysis helpers.
  No render target in a headless binary (they *could* emit an SVG
  string if the viz crates were linked; just not wired).
- **GPU `device(...)`, ports / applets / extensions, engram, LLM
  dispatch, tokenizers** -- runtime / connect surfaces.

## The plan

"Compile *all* MLPL" is not the target -- some surfaces (training,
GPU, a windowed viz) have no meaningful compiled form. The goal is
**compile everything that makes sense as a standalone CLI, and give a
precise error for the rest.** In priority order:

1. **Finish category A** (mechanical, high value): namespaces Phase 1b
   -> `du`/`StrList` -> `for`/`repeat` -> the remaining pure
   array/codec builtins. Each is a registry row + an `mlpl-rt`
   function + a gated e2e.
2. **Keep the boundary tested** -- this doc + the coverage gate, so
   interpreter-only is a visible, asserted list rather than a
   build-time surprise.
3. **Decide the ML story deliberately** -- either declare training
   interpreter/GPU-only (and make the compiler's error say so), or
   invest in a compiled *inference* path (forward-only `apply` of a
   loaded model -- far smaller than compiled training, and a natural
   pair with the pretrained-model extension). Recommendation: the
   former now, the latter only if a compiled-inference CLI is wanted.
4. **Optional: compiled SVG-string output** -- wire the viz crates
   into `mlpl-rt` if a compiled `dataflow`/`svg` that writes a file is
   useful. Low effort, on demand.
