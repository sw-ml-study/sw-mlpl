# Saga: max-elements

demo-algorithms / demo-ml-utils blocker: a cumulative collection
decode limit for parse_json / parse_toml, beside max_depth and
max_bytes. Bounds the TOTAL number of collection elements
(object fields + array cells + string-list items) a document may
produce, so a decoder can refuse an input that would allocate a
huge structure -- a semantic cap independent of byte size.

## Design

- Extend the parse-options record with `max_elements` (default
  unbounded): `parse_json(text, {max_elements: 1000})`, same for
  parse_toml.
- json_decode threads a remaining-elements budget (`&mut usize`)
  through value/object/array; each object field, array cell, and
  string-list item consumes one -- exhausting the budget is an
  err(...) (bad input data, not a hard error).
- toml_decode holds one budget across the whole document (values
  reuse the JSON value parser, so array/record elements in TOML
  values count too).
- A malformed `max_elements` (negative/non-integer) is a hard
  error, like the other options.

## Steps
1. max-elements -- thread the elements budget through
   json_decode + toml value parsing; read max_elements in the
   options; catalog signature + lang-ref + glossary. TDD
   (under/over the cap for JSON objects + arrays and TOML;
   default unbounded; bad option -> hard error).
2. close -- rebuild serve+pages+repl, deploy, connect smoke,
   wiki row, q-and-a, --done.
