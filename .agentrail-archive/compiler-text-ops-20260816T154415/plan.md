# Saga: compiler-text-ops

Finish the byte/text conversion rung of the compile-to-Rust path
(`mlpl-lower-rs` -> `mlpl_rt`), the last text feature before the
separate `compiler-process-semantics` saga (print / eprint / exit /
read_stdin + pristine stdout). The byte/bit I/O rungs (bit-ops,
byte-validator/stdout parity, read_bytes/file_size, write_bytes/
append_bytes) shipped in the prior `compiler-byte-io` saga; this saga
lowers the three string<->bytes<->int conversions so a standalone
compiled program can move between text, bytes, and integers with the
SAME semantics as the interpreter.

Builtins to lower (interpreter reference in parentheses):
- `tokenize_bytes(str) -> rank-1 byte array` -- UTF-8 bytes of the
  string as f64 cells in 0..=255 (mlpl-eval `tokenizer.rs`).
- `decode_bytes(byte array) -> str` -- inverse; MUST reuse the
  loud-reject array->bytes validator (rank <= 1, each cell an integer
  in 0..=255, else a descriptive error -- NEVER `x as u8`), then
  UTF-8-decode (invalid UTF-8 -> err) (mlpl-eval `tokenizer.rs`, the
  compiled validator added in compiler-byte-io step 2).
- `to_int(str) -> int Result` -- parse a string to an integer;
  parse failure is an `err` Result, not a panic (mlpl-eval
  `result_ops.rs`).

Two load-bearing parity rules (must match the interpreter exactly):
- `decode_bytes` LOUD-REJECTS non-integer / out-of-0..=255 cells via
  the shared validator; it does not truncate.
- `to_int` returns `CVal::Result` (ok(int) / err(msg)); a lowered call
  name alone is not acceptance -- a bad parse must produce an err the
  compiled program can branch on.

Compiler dispatch to extend: `mlpl-lower-rs/src/fncall.rs`. Runtime
home: a text module in `mlpl-rt-value` (or `mlpl-rt-fsio` if it needs
the byte validator already living there). `tokenize_bytes` /
`decode_bytes` take/produce `CVal`; `to_int` returns a `CVal::Result`.

Each step is TDD (RED failing test -> GREEN minimal lowering -> refactor)
with a gated end-to-end test compiling a tiny program and checking its
output. Hold or lower sw-checklist each step.

## Steps

1. tokenize-bytes -- lower `tokenize_bytes/1` (str -> rank-1 byte
   array). Add the runtime function (UTF-8 bytes -> CVal byte array)
   and the `fncall` lowering arm. TDD + gated e2e: a compiled program
   turns a string literal into its byte array and prints/round-trips
   the cells.

2. decode-bytes -- lower `decode_bytes/1` (byte array -> str), reusing
   the compiled loud-reject array->bytes validator from
   compiler-byte-io; invalid UTF-8 -> err. TDD + gated e2e: a compiled
   program round-trips str -> tokenize_bytes -> decode_bytes -> str
   and an out-of-range / non-integer cell yields an err (not a
   truncated decode).

3. to-int -- lower `to_int/1` (str -> int) returning a `CVal::Result`
   (ok(int) / err on parse failure). TDD + gated e2e: a compiled
   program parses a valid integer string to a number and a
   non-numeric string to an err the program branches on.

4. docs-close -- user + contract docs for the three conversions
   (lang-reference / compiler-implementation as appropriate, WHAT/HOW
   only), update the compiler-io-parity queue entry (text rung
   SHIPPED; compiler-process-semantics is next), wiki errata if a
   claim flips, and refresh the companion-demo-file-processing note if
   the text conversions clear a gate. Hold sw-checklist. `--done`.
