# Saga: range-read

demo-ml-utils blocker: large-file analysis (e.g. safetensors
headers) is gated by read_bytes materializing the WHOLE file.
Add bounded seek-based reads so a program can inspect a slice of
a multi-GB file without loading it.

## Design (extends fs_bytes.rs; sandboxed, Result-based)

- `read_bytes(path)` -- unchanged (whole file).
- `read_bytes(path, offset, length)` -- OVERLOAD: seek to
  `offset`, read at most `length` bytes (clamped at EOF), WITHOUT
  materializing the rest. ok(rank-1 byte array) / err. offset and
  length are non-negative integers; a bad type is a hard error;
  I/O outcomes are err Results.
- `file_size(path)` -- ok(byte count) from metadata (no read), so
  callers can bound and validate offsets.

Safetensors idiom this unlocks: read bytes 0..8 -> LE header
length N; read 8..8+N -> JSON header -> parse_json -> tensor
dtypes/shapes/offsets; seek to individual tensor ranges on
demand. No whole-file load.

Out of scope: a distinct typed byte-buffer type (that is
demo-memory packed-layouts; bytes stay f64 in 0..256).

## Steps
1. range-read -- read_bytes(path, offset, length) seek-based
   bounded read + file_size(path) in fs_bytes.rs; dispatch;
   catalog + lang-ref + glossary; TDD (range slice, EOF clamp,
   offset past end, whole-file still works, file_size, bad-type
   hard error, sandbox/no-sandbox err).
2. close -- rebuild serve+pages+repl, deploy, connect smoke,
   wiki row, q-and-a, --done.
