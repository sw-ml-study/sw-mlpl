# Saga: algebra-text-surface

Fix the sw-MLPL friction the demo-abstract-algebra dogfooding repo
found (its `docs/sw-mlpl-work-order.md`, verified against
`mlpl-repl 0.20.0`, build d373584c). Every item has a runnable repro +
acceptance tests in that file. These gaps are the INTERPRETER / CLI /
docs surface (NOT the compile-to-Rust path): strings cannot be joined
or built from numbers, and the documented bare-filename CLI
invocation fails. Order below is value-per-effort ("if you only do
three: A1, B1, B2").

Interpreter builtins live in `components/eval` (mlpl-eval, `fncall_*`
dispatch); the CLI is `components/cli` (mlpl-cli); docs are
`docs/lang-reference.md`. TDD + the acceptance cases from the work
order. Hold sw-checklist.

## Steps (recommended order)

1. cli-bare-filename -- A1: `mlpl-repl mini.mlpl` exits 1 because
   `Path::parent()` returns `Some("")` for a bare filename, so the FS
   sandbox root is the empty path and `FsProvider::new("")` fails.
   One-line fix in `mlpl-cli/src/include_script.rs` (treat an empty
   parent as absent, falling through to `.`) + a regression test.
   Acceptance: `mlpl-repl mini.mlpl` -> Ok, exit 0; ./ and sub/ forms
   unchanged.
2. str-concat-join -- B1: add `str_concat(a, b)` and
   `str_join(parts, sep)` interpreter builtins. Exact byte join,
   Unicode preserved, NO coercion (a number arg is an error, not a
   silent to_string); `str_join` is the linear-time fold (O(total),
   not O(n^2)). Deletes the downstream tokenize/decode bridge.
3. to-string -- B2: add `to_string(x)` -- number -> string, shortest
   round-trip (integral values bare: `to_string(8/2)` == "4", not
   "4.0"); the honest inverse of `to_number`. Removes the downstream
   dependency on `to_json`'s incidental scalar formatting.
4. string-list-u-arg -- B5: fix the domain check that rejects a
   string-list passed as a `u:` function argument (an oversight in
   one check; ~one line per the work order).
5. doc-svg-types -- A4: document all twelve `svg()` types in
   `docs/lang-reference.md` (add life / heatmap_grid / waffle /
   scatter3d / plotly3d / attention_overlay with their shapes) and
   correct or remove the `"hello " + name` `+`-concatenation example
   that does not run (it contradicts the no-string-`+` rule). Update
   the wiki errata if it tracks this.
6. close -- queue the remaining work-order items (B3
   str_len/str_slice/str_find/str_split, C1 labelled grid renderer,
   A2 recursion-depth cap, A3 + D1 comment-block rendering, the minor
   items) into `docs/future-sagas-queue.md`; --done.
