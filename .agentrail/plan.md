# readable-scripts

First saga of the Python-ML-developer ergonomics program
(docs/future-sagas-queue.md, approved 2026-09-23). Source: microgpt-mlpl
docs/sw-mlpl-requests.md #1 (formatting), #3 (destructuring), #6
(booleans); evidence 2,901 `str_concat(` calls and 269 nested
`str_concat(str_concat(` lines over 1,658 downstream `.mlpl` files.
Goal: a Python developer writes the obvious one-liner for output,
control-flow conditions and record unpacking.

Checked 2026-09-26: `and` / `or` / `not` appear in NO downstream `.mlpl`
as identifiers (only inside strings / comments), so they can become
keywords. The compile-to-Rust path (mlpl-lower-rs) dispatches from a
registry pinned by tests/dispatch_coverage_tests.rs.

## Steps

1. format-builtin -- `format(template, args...)` returning a string.
   Python format-spec subset: `{}` (auto-numbered), `{0}` (positional),
   width / fill / align `{:>8}` `{:<8}` `{:^8}` `{:08}`, integers
   `{:d}` `{:4d}` `{:x}` `{:b}`, floats `{:.4f}` `{:8.2f}` `{:e}` `{:g}`,
   `{{` / `}}` escapes; scalars and strings format, arrays render like
   `to_string` (or error for numeric specs -- decide + document). Errors
   name the bad field and argument index. Pure formatter module (no env),
   variadic dispatch. TDD against Python's outputs for a spec table.
2. write-and-variadic-concat -- `write(s)` writes a string to stdout
   with no newline (the missing sibling of `print`; replaces
   `unwrap(write_stdout(tokenize_bytes(s)))`), returns 0; `str_concat`
   accepts 2+ arguments. Both usable in script mode and connect mode.
3. boolean-operators -- `and` / `or` / `not` keywords: lexer tokens,
   parser precedence (below comparisons, `not` binds tighter than
   `and` binds tighter than `or`), eval: on scalars in `if` / `while`
   conditions they short-circuit; on arrays they are elementwise over
   0/1 masks (nonzero = true), result 0/1. Inside `grad` they are
   stop-gradient masks like the comparisons. TDD incl. short-circuit
   (the right side is not evaluated), precedence table, mask forms.
4. record-destructuring -- `{a, b} = expr` and `{a, b: x} = expr`
   (rename) and `{a, b} = expr?` as statements: parse a record pattern
   on the left of `=`, bind each field by name; a missing field is an
   error naming it and the available fields; extra fields are ignored.
   Works inside `u:` bodies (frame-scoped like any assignment) and on
   Results via `?`. Pure syntax: lowers to field reads.
5. compiler-parity -- lower `format`, `write`, variadic `str_concat`,
   `and` / `or` / `not`, and destructuring in mlpl-lower-rs where the
   compiler already lowers the surrounding forms (strings, records);
   anything it cannot lower yields a clear unsupported diagnostic.
   Extend dispatch_coverage_tests.
6. relay-close -- lang-reference (formatting spec table, booleans in
   the operator-precedence section, destructuring under assignment),
   glossary entries if the glossary covers operators, an example
   `examples/readable-scripts.mlpl` (docstrings + mlpl-fmt), emacs
   mlpl-mode keywords `and` / `or` / `not`, docs/downstream-updates.md
   (move these rows to shipped), q-and-a to microgpt-mlpl,
   future-sagas-queue, saga.md, CHANGES, wiki errata. Rebuild mlpl-repl
   release + debug. --done.
