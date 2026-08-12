# Saga: compiler-byte-io

Lower the byte / bit / text I/O builtins in the compile-to-Rust path
(`mlpl-lower-rs` -> `mlpl_rt`), matching interpreter semantics. The
compiler runtime (`mlpl-rt` / `mlpl-rt-value`) has NONE of this
surface today; each step adds runtime functions + `fncall` lowering +
tests. This unblocks demo-file-processing's real hexdump / histogram /
bit-field / format programs. Process semantics (print / eprint / exit
/ read_stdin) is the SEPARATE next saga (compiler-process-semantics).

Two load-bearing parity rules (interpreter, must match exactly):
- Byte writers LOUD-REJECT non-integer / out-of-0..=255 cells (mirror
  mlpl-eval `fs_bytes.rs` `array_to_bytes`: rank <= 1, each cell an
  integer in 0..=255 else error). NEVER `x as u8` truncation.
- Byte / file effects PROPAGATE I/O errors as `err` Results, never
  discard. The current compiled `write_stdout` does `let _ = ...` and
  truncates -- fix it.

Interpreter reference: dispatch `mlpl-eval/src/fncall_fs.rs`; impls
`fs_bytes.rs` / `fs_range.rs` / `fs_append.rs`; validator
`fs_bytes.rs:59-79`; bit ops in the pure `mlpl-runtime-bits` crate
(DenseArray->DenseArray, no Environment); text
`tokenizer.rs` (tokenize_bytes / decode_bytes), `result_ops.rs`
(to_int). Compiler dispatch to extend: `mlpl-lower-rs/src/fncall.rs`.

## Steps

1. bit-ops -- lower the pure bit-op family (band, bor, bxor, bnot,
   popcount, shl, shr, bmask, bits, from_bits) by depending on the
   pure `mlpl-runtime-bits` crate from `mlpl-rt` (re-export) and
   adding `fncall` arms. Pure DenseArray->DenseArray, elementwise +
   scalar broadcast; invalid-domain inputs are HARD errors (unwrap,
   matching interpreter RuntimeError, not an err Result). Golden e2e
   (compiled binary output): band(12,10)=8, bor(12,10)=14,
   bxor(12,10)=6, bnot(10,8)=245, shl(15,4,8)=240, shr(240,4)=15,
   from_bits(bits(165,8))=165, popcount(255)=8. Highest-value,
   lowest-risk slice: no Results, no I/O.

2. byte-validator-stdout-parity -- add a loud-reject array->bytes
   validator to the runtime (mirror `array_to_bytes`: rank <= 1, each
   cell an integer in 0..=255 else a descriptive error). Rewire
   compiled `write_stdout` to validate (reject, not truncate) and
   PROPAGATE write/flush errors as an `err` Result instead of the
   current discard. Closes both parity gaps for stdout; the validator
   is the foundation for write_bytes/append_bytes.

3. read-bytes-file-size -- add read_bytes/1 (whole file),
   read_bytes/3 (offset, length; non-negative integer args;
   EOF-clamped), file_size/1 to the runtime, returning CVal::Result
   (ok(rank-1 array) / ok(scalar) / err), path arg is a Str. DESIGN +
   implement the compiled sandbox-root mechanism (a compiled binary
   has no Environment; resolve the root from cwd or a build/runtime
   path, documented). Lower all three in `fncall`. e2e: write a temp
   file, read its bytes + size back through a compiled program.

4. write-append-bytes -- add write_bytes/2 + append_bytes/2 to the
   runtime (using the step-2 validator), returning ok(count) / err
   and propagating I/O errors. Lower them. e2e: compiled program
   writes bytes to a temp file, reads them back (round-trip), and an
   out-of-range byte value yields an err (not a truncated write).

5. text-ops -- lower tokenize_bytes/1 (str -> byte array),
   decode_bytes/1 (byte array -> str, shared validator), to_int/1
   (str -> int, parse failure -> err). Add runtime funcs. e2e: a
   str -> bytes -> str round-trip through a compiled program.

6. close -- update docs/companion-demo-file-processing.md (byte / bit
   / text gate cleared; remaining compiler gate = process-semantics),
   the docs/downstream-unblock-plan.md checklist, and
   docs/future-sagas-queue.md; resolve any wiki errata whose claim
   flips; --done.
