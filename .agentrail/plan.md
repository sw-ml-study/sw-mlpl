# Saga: record-keys

demo-ml-utils step 004 blockers (safetensors tensor-name
discovery + duplicate validation):

1. MLPL cannot enumerate a parsed record's keys -> arbitrary
   tensor names can't be discovered.
2. parse_json silently overwrites duplicate object members,
   losing the evidence needed to reject duplicate tensor names.

## Design

- `record_keys(record)` -> string-list of the record's keys in
  DETERMINISTIC (sorted) order -- records are BTreeMap-backed, so
  keys are already sorted. A non-record argument is a hard error
  (a type error, like the record accessors). Lives in a small
  dedicated module so the record-access module stays within its
  function budget.
- Duplicate-key rejection in parse_json: json_decode's object
  builder errors on a repeated key (`err(...)` with the byte
  position) instead of last-wins overwrite. TOML is out of scope
  here (the stated blocker is parse_json).

## Steps
1. record-keys -- record_keys builtin (new fncall_record_keys.rs,
   wired into eval_fncalls) + duplicate-key rejection in
   json_decode::object; catalog + lang-ref + glossary; TDD
   (sorted keys, empty record, non-record hard error; duplicate
   JSON key -> err, distinct keys still ok).
2. close -- rebuild serve+pages+repl, deploy, connect smoke,
   wiki row, q-and-a, --done.
