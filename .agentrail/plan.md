# Saga: type-detection

demo-algorithms serialization blocker: non-record root type
detection. Expose the internal `value_kind` as a `type_of(v)`
builtin returning a stable kind string ("array", "string",
"record", "result", "string-list", "model", "tokenizer",
"gen-state", "partial", "builtin-ref", "user-fn-ref",
"device-tensor"). This lets a program branch on a value's kind
at the root before calling kind-specific accessors
(has_field/record_get on records, to_json on data). Pure, total
(never errors), works on ANY value. Home: fncall_values.rs
(alongside repr/equal). Catalog/lang-ref/glossary; TDD.

## Steps
1. type-of -- eval_type_of in fncall_values.rs routing through
   mlpl_eval_types::value_kind; catalog/lang-ref/glossary;
   TDD (one assertion per kind + totality).
2. close -- rebuild pages+deploy, connect smoke of type_of,
   wiki row, q-and-a (demo-algorithms: type detection done;
   remaining byte-I/O/TOML/atomic/limits/streaming), --done.
