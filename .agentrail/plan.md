# Saga: toml-codec

demo-algorithms serialization: a TOML codec pair mirroring the
JSON codec, for config-driven demos. Hand-rolled (no serde/toml
crate), Result-based like to_json / parse_json, living beside the
JSON codec in mlpl-eval.

## Data model mapping (TOML root is always a table)

- `to_toml(record)` -> `ok(toml_text)` / `err`. Root MUST be a
  Record (TOML documents are tables) -- a non-record root is
  `err`. Fields emit SORTED (deterministic): scalar/string/array
  fields as `key = value` first, then nested Record fields as
  `[section]` / `[section.sub]` tables (recursive). Non-finite
  number -> `err` (as to_json). ok(x)/err(e) values -> err (not
  representable as TOML config).
- `parse_toml(text)` -> `ok(record)` / `err(message with line)`.

## Supported subset (documented boundary, present-tense)

Supported: comments (`# ...`), blank lines, bare keys
(`[A-Za-z0-9_-]+`), `key = value`, `[table]` and dotted
`[table.sub]` headers, values = integer, float, boolean
(`true`/`false` -> 1/0, as JSON), basic string `"..."` (JSON-
style escapes), homogeneous array `[...]` of numbers or of
strings.

Not supported (err, named): inline tables `{}`, arrays of tables
`[[...]]`, literal/multiline strings, datetimes, dotted-key
assignments (`a.b = 1`), and numeric bases/underscores. A byte is
still an f64; there is no TOML integer type distinct from float
beyond the printed form.

## Modules (mirror the json_* layout in mlpl-eval)

- `toml_encode.rs` -- record -> TOML text (sorted; sections).
- `toml_decode.rs` -- TOML subset -> record (line scanner +
  value parser); split a `toml_scalar.rs` if function budget
  requires.
- `fncall_toml.rs` -- try_dispatch + eval_to_toml + eval_parse_toml
  wrapping the codec in ok/err Results; wired into eval_fncalls.

## Steps
1. to-toml -- toml_encode.rs + fncall_toml.rs (to_toml only);
   catalog/lang-ref/glossary; TDD (scalars, strings, arrays,
   sorted keys, nested sections, non-record/non-finite/result
   -> err).
2. parse-toml -- toml_decode.rs (+ toml_scalar.rs if needed);
   wire eval_parse_toml; TDD (kv, tables, dotted, arrays,
   booleans, comments, malformed -> err with line) + round trip
   with to_toml.
3. close -- rebuild serve+pages+repl, deploy, connect smoke,
   wiki row, q-and-a (TOML done; remaining: decode limits, typed
   native, streaming, higher-rank JSON), --done.
