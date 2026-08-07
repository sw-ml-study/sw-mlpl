# Saga: mlplunit-round-3
Their structured-event-consumption gate: parse_json(s) --
JSON text to a typed MLPL record (the inverse of the event
encoder). Mapping: object -> record, string -> Str, number ->
scalar, homogeneous number array -> vector, homogeneous string
array -> string list, bool -> 1/0, null -> zilde; nested
objects recurse; malformed JSON -> err with position.
## Steps
1. parse-json -- decoder + builtin, TDD (their fixture shape,
   unicode exactness, round-trip with the event encoder,
   malformed diagnostics); docs rows; run their gate.
2. close -- rebuilds/deploy, wiki, q-and-a, --done.
