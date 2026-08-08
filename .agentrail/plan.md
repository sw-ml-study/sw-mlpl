# Saga: byte-io

demo-algorithms serialization blocker: raw bytes / I/O. Two
sandboxed fs endpoints so binary formats round-trip:

- `read_bytes(path)` -> `ok(bytes)` / `err(msg)`: a rank-1 array
  of byte values (0..256), sandbox-contained like read_text.
- `write_bytes(path, bytes)` -> `ok(1)` / `err(msg)`: writes a
  rank-<=1 byte array; each cell must be an integer 0..=255
  (hard error naming the offending cell, same validation as
  decode_bytes); a non-array byte arg is a hard error.

A byte is an f64 in 0..256 -- the convention already set by the
bit ops and tokenize_bytes/decode_bytes. String<->bytes is
ALREADY done (tokenize_bytes/decode_bytes), so the full binary
serialization round trip composes without new codec surface:

    write_bytes(p, tokenize_bytes(to_json(v)))     # encode + store
    parse_json(decode_bytes(read_bytes(p)))        # load + decode

Placement: a new fs_bytes.rs module in mlpl-eval (fs ops need
Environment.fs_root; native + connect; the browser has no
sandbox so it returns the same "no filesystem sandbox" err as
read_text). Shares one array->Vec<u8> validator (array_to_bytes)
with decode_bytes -- define once, invoke twice. Keeps fs_ops.rs
at its function budget by living in a sibling module.

## Steps
1. byte-io -- fs_bytes.rs (read_bytes/write_bytes + shared
   array_to_bytes; refactor decode_bytes to share it); dispatch
   in fncall_fs; catalog/lang-ref/glossary; TDD (round trip,
   missing file err, domain errors, sandbox escape, full
   to_json/parse_json binary compose).
2. close -- rebuild pages+deploy, connect smoke, wiki row,
   q-and-a (byte I/O done; demo-algorithms remaining: TOML +
   native codecs, atomic writes, decode limits, streaming),
   --done.
