# Saga: decode-limits

demo-algorithms serialization: decode limits (size + depth caps)
so parse_json / parse_toml are safe on adversarial or untrusted
input. The recursive-descent JSON decoder can stack-overflow on
deeply nested objects (`{"a":{"a":{"a":...}}}`); a depth budget
prevents it, and a byte cap rejects oversized input before
parsing.

## Design

- A shared `Limits { max_depth, max_bytes }` (decode_limits.rs)
  with a sensible default (`max_depth = 128`, matching
  serde_json; no byte cap by default). Depth is ALWAYS enforced
  (the safety guard); a caller can override.
- Optional 2nd argument: an options record.
  `parse_json(text, {max_depth: N, max_bytes: N})` and the same
  for `parse_toml`. One-arg calls keep today's behavior (default
  limits) -- backward compatible.
- json_decode threads a remaining-depth budget through
  value/object/array; a container opened at depth 0 is an
  err(...). decode checks `text.len() > max_bytes` first.
- toml_decode enforces the same byte cap and passes max_depth to
  `toml_scalar::parse_value` (TOML RHS values reuse the JSON
  value parser, so nested-array depth is covered).
- Bad options (non-integer / negative max_depth or max_bytes,
  or a non-record 2nd arg) are a loud error.

## Steps
1. decode-limits -- decode_limits.rs (Limits + option parsing);
   thread depth through json_decode; byte cap in json_decode +
   toml_decode; parse_value takes max_depth; eval_parse_json /
   eval_parse_toml read the optional options record; catalog
   signatures + lang-ref + glossary. TDD (default ok, depth
   exceed -> err, byte cap -> err, bad options -> err, both
   codecs).
2. close -- rebuild serve+pages+repl, deploy, connect smoke,
   wiki row, q-and-a (decode limits done; remaining: typed
   native formats, streaming, higher-rank JSON), --done.
