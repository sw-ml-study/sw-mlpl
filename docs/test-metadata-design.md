# Test metadata and reflection: design

Status: DRAFT for review (saga mlplunit-unblock, step 014).
Upstream contract: mlplunit's sw-MLPL-changes-needed.md item 4
(acceptance fixture `tests/capabilities/metadata_case.mlpl`,
which uses a bare `@test` line before `def u:...`).

## Surface

```text
@test
def u:addition_works() { u:assert_eq(2 + 2, 4, "adds")? ; ok(1) }

@test {name: "parses cleanly", tags: ["fast", "parser"], skip: "flaky on wasm"}
def u:parse_case() { ... }
```

- `@test` is an ANNOTATION line attached to the NEXT `def u:`
  statement. The `@` character is unclaimed today (it lexes as
  UnexpectedCharacter), so the syntax costs nothing existing.
- The optional argument is one RECORD LITERAL -- reusing the
  existing record grammar, no new metadata mini-language.
  Recognized fields, all optional:
  - `name`: stable test name (string). Default: the function's
    own name without the `u:` prefix.
  - `tags`: a string list.
  - `skip`: a string reason. Present = skip (the reason is the
    documentation).
  - `expected_failure`: scalar 1 -- the runner treats a failing
    result as pass-inverted.
  - `timeout_ms`: scalar. RECORDED, not enforced -- enforcement
    is the runner's job (contract item 7 owns process controls);
    stated plainly in docs.
  Unknown fields are a structured error (malformed metadata must
  not pass silently).
- `@test` on anything other than a following `def u:` statement
  is a structured parse-time error; two annotations in a row
  likewise.

## Reflection (discovery without execution)

Two builtins, sidestepping the ordered-collection gap (records
sort by key; the contract demands SOURCE order):

| Builtin | Semantics |
|---|---|
| `tests()` | The stable test names as a string list, in definition (source) order -- across `include` chunks in splice order. |
| `test_info("name")` | A record for one test: `{name, fn, tags, skip, expected_failure, timeout_ms, source, line}` -- `fn` is the `:u:name` REFERENCE (so a runner does `call(test_info(n).fn)`), `source`/`line` locate the definition. Absent optional fields default (`tags` empty list, `skip` empty string, `expected_failure` 0, `timeout_ms` 0). |

Discovery has no side effects: evaluating `def` statements
registers tests (that is how definitions already work);
enumerating via `tests()` / `test_info` never invokes anything.
A discovery run is `mlpl-repl --source-dir ... -f suite.mlpl`
where the suite contains only `include` + definitions, then the
runner asks `tests()`.

## Semantics decisions

- Registration happens when the annotated `def` EVALUATES (same
  moment the function itself becomes callable) -- include order
  therefore IS registration order, deterministically.
- Duplicate stable names: the second registration is a
  structured eval error naming both definitions' sources. Names
  are the runner's identity; collisions must be loud.
- Re-defining an annotated function replaces its registration in
  place (keeps its original order slot), consistent with late
  binding of references.
- `:describe u:name` gains a metadata line when present;
  `:fns` marks annotated functions with their test name.
  (Nice-to-have; rides the implementation step if budget holds.)
- Source attribution: the script runner already knows each
  chunk's source id; it sets a current-source display name on
  the environment before evaluating a chunk, and `def`
  registration stamps `{source, line}` from it plus the def's
  span. Non-script surfaces stamp `repl`.

## Implementation steps

- metadata-parser -- lexer claims `@`; parser parses
  `@test [record-literal]` and attaches it to the following
  `def` (AST: `FnDef` gains an optional `meta` record expr);
  structured errors for stray/duplicated annotations. TDD.
- metadata-registry -- `UserFn` carries the evaluated metadata +
  `{source, line}`; a source-ordered registry list in the env;
  duplicate-name diagnostics; current-source stamping from the
  chunked script runner. TDD.
- metadata-reflect -- `tests()` + `test_info()` builtins, docs
  (usage, lang-reference, glossary, catalog), `:describe`/`:fns`
  riders if budget holds; run mlplunit check-capabilities
  expecting test-metadata to flip AVAILABLE.

## Open questions for review

1. Annotation spelling: exactly `@test` (proposed, matches the
   fixture) -- or a general `@meta` the test runner interprets?
   Proposed: `@test` only, and reject other `@words` with a
   structured error naming the one supported annotation;
   generality can come when a second consumer exists.
2. `skip` semantics: sw-MLPL records the reason and the runner
   decides (proposed), vs `call` refusing to invoke skipped
   tests? Proposed: recording only -- the language should not
   police what a runner may deliberately do (e.g. --run-skipped).
3. The registry surface: two builtins (`tests()` +
   `test_info(name)`, proposed) vs one `tests()` returning a
   record keyed by name (loses source order) vs a record with a
   `names` string-list field plus per-test sub-records
   (single call, slightly awkward)? Proposed: the two-builtin
   form for clean ordering plus simple per-test access.
