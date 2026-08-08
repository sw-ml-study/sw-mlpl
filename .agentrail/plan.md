# Saga: serialization-results

Make the serialization primitives Result-based and honest, per
the user's decision (2026-08-07). Two shipped behaviors change:

1. `to_json(v)` currently returns a BARE string and hard-errors
   on a non-data kind -- and it EMITS INVALID JSON for non-finite
   numbers (`to_json(1/0)` -> `inf`, `to_json(sqrt(-1))` -> `NaN`,
   which no conforming parser accepts). Change it to return
   `ok(json_string)` / `err(message)` -- consistent with
   parse_json and the fs I/O layer -- and treat NaN / +-Inf as an
   `err` (JSON has no non-finite numbers). Non-data kinds also
   become `err` rather than a hard error.

2. `write_bytes` / `write_atomic` currently HARD-ERROR on invalid
   input (out-of-range/non-integer byte cell, or a wrong-typed
   value). Change these to `err(...)` Results so downstream can
   catch them as data, matching the rest of the fs API. The
   shared `array_to_bytes` validator stays hard-erroring
   internally (decode_bytes, a pure tokenizer transform, keeps
   its loud contract); write_bytes/write_atomic CATCH it and
   surface an `err` Result.

Deferred (separate sagas / declined): distinct byte-buffer type
-> demo-memory packed-layouts; higher-rank & Result JSON round-
trip semantics, decode limits, TOML, typed native formats,
streaming -> later sagas.

## Steps
1. to-json-result -- to_json returns ok/err; non-finite -> err
   (inline finite guard in json_encode, no new function to stay
   under the module budget); non-data kind -> err. Update
   fncall_json, to_json_tests, fs_bytes/fs_atomic compose sites
   (unwrap to_json), lang-ref, glossary. TDD.
2. byte-write-result -- write_bytes + write_atomic input
   validation (range + wrong-type) -> err Result. Update
   fs_bytes.rs, fs_atomic.rs, their tests, lang-ref, glossary.
   TDD.
3. close -- rebuild serve+pages+repl (demo-algorithms reads
   target/release/mlpl-repl), deploy, connect smoke, wiki row,
   q-and-a (Result contract; what demo-algorithms can treat as
   unblocked vs change), --done.
