# Saga: native-codec

demo-algorithms blocker (docs/sw-mlpl-blocker.md): a typed-native
binary value codec. to_native(value[, opts]) -> ok(bytes)/err and
parse_native(bytes[, limits]) -> ok(value)/err -- a versioned,
self-describing, resource-bounded binary format that losslessly
round-trips every MLPL data value, Result-based like the JSON
codec. Composable: no runtime value-model change; bytes are the
existing f64-array convention, so write_bytes(unwrap(to_native(v)))
and parse_native(unwrap(read_bytes(p))) compose directly.

## Wire format v1 (canonical little-endian)
Header: magic "MLPB" (4) + version=1 (1 byte) + payload_len u32 LE.
Then one tagged value (recursive), tag byte:
  0 scalar: f64 LE (8)
  1 array:  u8 rank + rank*u32 LE dims + prod*f64 LE data
  2 string: u32 LE len + UTF-8
  3 strlist: u32 count + [u32 len + UTF-8]*
  4 record: u32 fields + [u32 keylen + key + value]* (sorted keys)
  5 result-ok: value ; 6 result-err: value
Non-data kinds (model/tokenizer/gen-state/partial/refs/device)
-> err, never partial. Deterministic (equal value -> equal bytes;
records BTreeMap-sorted). All numbers f64 (documented loss policy:
one numeric element type). Integrity: payload_len validated on
decode (reject truncation/oversize); stronger checksum + cycles/
references are documented follow-ups.
Decoder enforces limits (reuse decode_limits Limits: max_bytes up
front, max_depth threaded, max_elements post-check) BEFORE
allocating; malformed/truncated/oversized/too-deep/unsupported
-> err with a field/index path where feasible.

## Modules (mirror json_* layout in mlpl-eval)
- native_encode.rs (Value -> Vec<u8>), native_decode.rs
  (bytes -> Value, budget-checked), fncall_native.rs
  (to_native/parse_native, ok/err wrapping), wired in eval_fncalls.

## Steps
1. native-encode -- to_native + wire header + tagged encode; TDD
   (each kind, deterministic bytes, non-data -> err, write_bytes
   compose).
2. native-decode -- parse_native: header/version validate,
   budget-bounded recursive decode, round trip + corrupt/
   truncated/bad-magic/bad-version/over-budget -> err + golden
   bytes; catalog+lang-ref+glossary.
3. close -- rebuild serve+pages+repl, deploy, connect smoke, wiki
   row, q-and-a (demo-algorithms can proceed), --done.
