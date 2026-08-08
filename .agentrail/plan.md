# Saga: atomic-writes

demo-algorithms serialization blocker: atomic writes. Today
write_text / write_bytes can leave a half-written file if the
process dies mid-write; persist-to-disk demos need crash-safe
writes. One new sandboxed builtin:

- `write_atomic(path, value)` -> `ok(1)` / `err(msg)`: `value` a
  string (UTF-8 bytes) or a rank-<=1 byte array (0..=255).
  Writes to a hidden temp file in the SAME directory, then
  `rename`s it over the target -- atomic on POSIX same-fs, so a
  reader sees either the old file or the whole new one, never a
  torn write. On failure the temp is cleaned up. Sandbox-
  contained like the other fs ops; a non-string/non-array value
  is a hard error.

It leans on the type distinction already exposed by type_of
(string vs array) and reuses fs_bytes::array_to_bytes for byte
validation, so `write_atomic(p, to_json(v))` and
`write_atomic(p, tokenize_bytes(data))` both work. Lives in a
new fs_atomic.rs (keeps fs_ops.rs / fs_bytes.rs at their
function budgets), wired through fncall_fs dispatch.

## Steps
1. atomic-writes -- fs_atomic.rs (write_atomic: string|byte
   array, temp + rename, temp cleanup on error); dispatch in
   fncall_fs; catalog/lang-ref/glossary; TDD (string round trip,
   byte round trip, overwrite, no temp left behind, domain +
   sandbox errors, to_json compose).
2. close -- rebuild serve+pages AND target/release/mlpl-repl
   (demo-algorithms reads ../sw-mlpl/target/release/mlpl-repl),
   deploy, connect smoke, wiki row, q-and-a (atomic writes done;
   remaining: TOML/native codecs, decode limits, streaming),
   --done.
