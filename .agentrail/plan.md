# Saga: to-json
demo-algorithms serialization blocker (highest value, encoder
already exists internally as event_json). to_json(value) ->
JSON string: the deterministic encode half completing the
parse_json<->to_json round trip. Objects=records (sorted keys),
arrays=flat/nested by shape, strings escaped exact, numbers
bare-int/float, Result -> {ok, value|error}; non-data kinds
(model/tokenizer/gen-state/partial/refs/device-tensor) error
loudly. Share the escaper/number helpers with event_json.
## Steps
1. to-json -- json_encode.rs general encoder + to_json builtin;
   event_json shares helpers; catalog/lang-ref/glossary; TDD.
2. close -- rebuild pages+deploy, connect smoke, wiki,
   q-and-a (round-trip; demo-algorithms re-scope), --done.
