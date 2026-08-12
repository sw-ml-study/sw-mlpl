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

1. **Source loading** -- RESOLVED (compiler-source-loading, shipped
   2026-08-10): `mlpl-build` now resolves `include` via the
   interpreter's `expand()` + a filesystem sandbox and lowers the
   flattened AST directly, with `--source-dir`. The real
   hexdump/histogram/WAV/Ogg programs now get PAST the include gate
   to the next wall (`FnDef`).
2. **User functions + control flow + records** -- PARTLY RESOLVED
   (compiler-functions param-only + compiler-control-flow, shipped
   2026-08-11): `def u:` lowers to a nested Rust fn over its
   parameters; `if/else` lowers to a Rust if-expression and `while`
   to a Rust `while` over mutable bindings (first `=` -> `let mut`,
   rebind -> reassign). Record literals `{a: 1, b: 2}` lower to
   `CVal::Record`, `r.a` field access unwraps a numeric field, and
   `ok(x)`/`err(x)` build `CVal::Result`. A function whose body
   produces `ok`/`err`/a record (or uses `?`) now lowers to
   `-> CVal`; `?` (`check`) unwraps an `ok` payload or early-`return`s
   the `err`. The full demo pattern compiles and runs:
   `def u:run(n) { f = u:fit(n)?; f.slope }` where `u:fit` returns
   `ok({...})`. Still rejected: `for`; record-of-record /
   string-leaf field chains; arithmetic directly on a `?`-unwrapped
   value (unwrap-then-field-access is the supported shape);
   top-level `?` (outside a `def u:`); and functions that read a
   global. The value-model gate is CLEARED -- the remaining compiler
   gates are byte/bit lowering and process semantics.
3. **Shared byte validation + error propagation** -- PARTLY
   RESOLVED (byte-validator-stdout-parity): the compiler runtime now
   has a loud-reject `array_to_bytes` (rank <= 1, each cell an
   integer in `0..=255` else a descriptive error, mirroring
   mlpl-eval `fs_bytes.rs`), and compiled `write_stdout` validates
   (rejects, never `as u8` truncation) and returns `ok(count)` /
   `err(msg)`, propagating write/flush failures instead of discarding
   them. `write_bytes` / `append_bytes` will reuse the same
   validator. Interpreter and compiled `write_stdout` now share
   semantics.
4. **Byte / bit / text lowering** -- the bit ops (`band` etc.) now
   lower; `read_bytes/1`, `read_bytes/3`, `write_bytes`,
   `append_bytes`, and the byte/text conversions are the remaining
   rungs.
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

## Second gate: binary source + persistent handles

Distinct from the compiler track and from in-memory codec
chunking. The runtime does not yet provide binary stdin, explicit
backpressure, or consumable source/sink HANDLES. Transcoding from
seekable files can proceed once codec extensions exist, but true
stdin-driven streaming needs these effects. The ask is
COMPOSITIONAL, not codec-specific:

- A sink handle: bounded writes, partial-write handling,
  flush/close cleanup, sandboxing, offsets beyond f64 integer
  ambiguity, and a memory high-water invariant (resident memory
  proportional to chunk + writer state, not total output).
- A later sequential SOURCE handle: explicit EOF, backpressure,
  and matching error/lifecycle semantics; must reproduce the
  accepted range-reader results across split fields.

Already delivered on the sandboxed file-PATH subset (no handle):
`append_bytes(path, bytes) -> ok(count)` (bounded per-call
append, implicit close/flush) and `write_stdout(bytes) ->
ok(count)` (ordered per-call-flushed binary stdout). These cover
interpreter-side seekable-input-to-file-or-stdout; they do NOT
satisfy binary stdin, persistent handles, or explicit
backpressure. Sink acceptance: byte-identical output at chunk
sizes 1/7/64/65536, injected partial/failed writes, output above
the memory budget, resident memory bounded by chunk + writer
state.

## Third gate: authorized codec extensions

Seekable-file transcoding is gated on codec extensions existing --
a codec provided as an AUTHORIZED extension. This rides the
extension track (the registry + C-ABI adapter have shipped; a
trust/authorization resolver and dynamic load remain -- see the
`extensions-*` queue entries), not a set of format-specific
builtins. Explicitly: do NOT add MP3/Ogg/WAV builtins upstream.

## Deferred (their call, not a blocker)

Packed bytes / typed `u8` arrays: reads currently allocate one
f64 per byte. Inefficient but not blocking (fixed chunking already
bounds file-copy RSS). They will request a packed representation
when density/throughput is a concrete target. Maps to the
`codec-numeric-types` queue item.

## Mapping to sw-mlpl work

This is exactly the **phased compiler expansion** track (the
compile-to-Rust `CVal` groundwork shipped as Saga A: strings +
`write_stdout`/`args`/`arg`). The rungs above become the next
compiler sagas -- see `docs/future-sagas-queue.md`
(`compiler-*` rungs). It is independent of the extension work.
