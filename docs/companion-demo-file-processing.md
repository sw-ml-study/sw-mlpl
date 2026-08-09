# Companion contract: ../demo-file-processing

Source: `../demo-file-processing/docs/upstream-contract.md` (and
the acceptance reports it links). This repo does not modify
sw-mlpl; each request is the smallest general capability motivated
by an executable probe. Recorded 2026-08-09.

## Already delivered (no upstream request outstanding)

- **Bounded range reads**: `read_bytes(path, offset, length) ->
  Result<Array, Error>` and `file_size(path) -> Result<Number,
  Error>` (middle ranges, EOF clamping, beyond-EOF empty,
  zero-length verified). Unblocks in-memory hexdump / histogram /
  bitfield / WAV / MP3-ID3 / Ogg inspection -- all confirmed to
  need NO format-specific upstream API (bounded reads + MLPL state
  are the accepted downstream contract).
- **Numeric `mlpl-build` parity**: the compiler lowers the
  arithmetic probe (`reduce_add((iota(8)+1)*2)` -> `72`), matching
  the interpreter. (This is the Saga-A compile-to-Rust groundwork.)

## The central blocker: compiled-application runtime parity

Every remaining file/format demo compiles the SAME MLPL algorithm
that already runs in the interpreter, then fails in `mlpl-build`
lowering. No new codec builtin is requested -- the generic compiler
surface is the gate. The reports order the missing rungs:

1. **Source loading** -- `include` directives are unresolved by the
   compiler (earliest gate; the real hexdump/histogram/WAV/Ogg
   programs all fail here first).
2. **User functions + control flow** -- `FnDef` is unsupported in
   lowering; also Results and records.
3. **Shared byte validation + error propagation** -- compiled
   invalid bytes are coerced rather than rejected; runtime write
   errors are discarded. Interpreter and compiled paths must share
   semantics, not just share a call name.
4. **Byte / bit / text lowering** -- `read_bytes/1`,
   `read_bytes/3`, `write_bytes`, `band/2` (and the bit ops) are
   rejected during lowering.
5. **Process entry / status semantics** -- `read_stdin`, `print`,
   `eprint`, `exit` are not lowered; the current `write_stdout`
   wrapper also appends a spurious textual result line after binary
   stdout.
6. Then: positive byte + format artifact parity, plus a repeated
   clean-environment (source-free) audit.

Minimum acceptance (interpreter and compiled using identical
semantics): `args` / `print` / `eprint` / `exit`; `read_bytes`
(whole + range); `file_size` / `write_bytes`; bit ops; `include`;
`FnDef` + control flow + Results/records.

Standalone artifacts already run source-free with no named
parser/REPL/evaluator dependency, so the blocker is application
COVERAGE + SEMANTICS in the compiler, not the artifact-launch
mechanism.

## Mapping to sw-mlpl work

This is exactly the **phased compiler expansion** track (the
compile-to-Rust `CVal` groundwork shipped as Saga A: strings +
`write_stdout`/`args`/`arg`). The rungs above become the next
compiler sagas -- see `docs/future-sagas-queue.md`
(`compiler-*` rungs). It is independent of the extension work.
