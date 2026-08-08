# Saga: tagged-envelope

Implements the reserved `$mlpl` tagged-envelope encoding recorded
in `docs/serialization-variant-encoding.md`. One mechanism closes
three open serialization items: higher-rank text round-trips, a
shape/type envelope, and the text half of typed-native encoding.

## Envelope shape (inner keys sort deterministically)

    {"$mlpl":{"data":[0,1,2,3],"shape":[2,2],"type":"array","v":1}}
    {"$mlpl":{"type":"result","v":1,"value":5,"variant":"ok"}}
    {"$mlpl":{"error":"boom","type":"result","v":1,"variant":"err"}}
    {"$mlpl":{"fields":{...},"type":"record","v":1}}   # $mlpl-key escape

## Design

- ENCODE: `to_json(v, {tagged: 1})`. A pure transform `wrap(v)`
  rewrites the value into a plain-JSON-representable form where a
  rank->=2 array becomes an {shape, data} envelope, a Result a
  {variant, value|error} envelope, and a record that literally
  contains a `$mlpl` key is escaped as a {fields} envelope;
  everything else (records, rank-<=1 arrays, strings, string
  lists, scalars) passes through, recursively. Then the EXISTING
  sorted-key encoder serializes it -- no new hand-written JSON,
  deterministic output for free.
- DECODE: `parse_json` reconstructs `$mlpl` envelopes
  UNCONDITIONALLY (the reserved key is never application data, so
  no opt-in is needed -- unlike the compact {results:1} form,
  which stays for plain-JSON interop). A post-decode walker
  `unwrap_envelopes(v)` rebuilds the array/result/record.
- So to_json(v, {tagged:1}) <-> parse_json round-trips ANY data
  value losslessly, including rank->=2 arrays and Results.

Out of scope: a general user-defined variant type (none exists;
the envelope's `type:"option"` etc. wait for that language
decision). TOML tagged mode (JSON first).

## Steps
1. envelope-encode -- envelope.rs `wrap`; to_json {tagged:1}
   option; catalog/lang-ref/glossary; TDD (rank>=2 array
   envelope, ok/err envelope, nested, $mlpl-key escape, plain
   values unchanged).
2. envelope-decode -- `unwrap_envelopes`; parse_json calls it
   unconditionally; TDD (round-trip rank>=2 + Result + nested +
   escape; plain JSON unaffected; malformed envelope tolerated).
3. close -- rebuild serve+pages+repl, deploy, connect smoke,
   wiki row, q-and-a, --done.
