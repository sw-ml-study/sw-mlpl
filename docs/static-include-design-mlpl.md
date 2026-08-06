# Static include: the sw-MLPL loader design

Status: DRAFT for review (saga mlplunit-unblock, step 003).
Upstream contract: mlplunit's `docs/static-include-design.md`
(the executable acceptance spec is its
`tests/native_include/native_include_case.mlpl`, gated behind
`scripts/verify-native-include`).

## Surface

```text
include "vector.mlpl"
```

- Top-level DECLARATION only: legal exactly at statement
  position of a source file, rejected inside blocks, function
  bodies, or expressions.
- The argument is one literal relative path -- never computed,
  conditional, absolute, or remote.
- `include` stays a CONTEXTUAL identifier, not a reserved
  keyword: the parser recognizes `Ident("include")` followed
  immediately by a string literal at statement position (that
  token sequence is a parse error today, so claiming it breaks
  nothing, and `include = 5` keeps working -- same technique as
  the keyword-field-names fix). A trailing `;` is accepted as
  the ordinary statement separator.
- Included definitions (u: functions, bindings) enter the
  program at the include SITE, in source order; existing
  duplicate-definition rules apply unchanged.

## The chunked-program model (no span surgery)

`mlpl_core::Span` is `{start, end}` byte offsets into ONE
source. Rather than threading a `source_id` through every span
(broad churn) or flattening text before lexing (explicitly
rejected upstream -- it destroys diagnostics), the loader keeps
files separate end to end:

1. Each file is lexed and parsed INDEPENDENTLY -- its spans stay
   valid byte offsets into its own text.
2. The loader recursively expands `Include` nodes into an
   ordered `Vec<Chunk>` where
   `Chunk = { source: SourceId, stmts: Vec<Expr> }`, splicing
   included chunks at the include site.
3. A `SourceTable` maps `SourceId -> (display path, text)`.
4. The script runner evaluates chunks IN ORDER in one shared
   Environment. The evaluator itself is untouched and stays
   filesystem-free.
5. Any parse or eval diagnostic is rendered against the chunk's
   own source: `path:line:column` computed from the offset and
   that file's text. Errors in included files name the included
   file, not the root.

Final-value / exit-code semantics are unchanged: chunks splice
at the include site, so the root file's last statement is still
the program's last statement.

## Resolution and safety (native)

- `mlpl-repl` gains `--source-dir DIR` (independent of
  `--data-dir`); when absent it defaults to the directory of the
  root script.
- Resolution: the include path must be relative; it joins the
  INCLUDING file's directory, canonicalizes, and must remain
  under the canonical source root -- absolute paths, `..`
  escapes, and symlink escapes all reject with a structured
  error naming the offending path and the root.
- Load-once: a canonical file expands once per program;
  repeated includes are idempotent no-ops.
- Cycles: the loader keeps the active include stack; a repeat
  appearance errors with the COMPLETE chain
  (`a.mlpl -> b.mlpl -> a.mlpl`).

## The provider seam

```text
trait SourceProvider {
    fn resolve(&self, from: &SourceId, rel: &str) -> Result<SourceId, IncludeError>;
    fn read(&self, id: &SourceId) -> Result<String, IncludeError>;
}
```

- Native: `FsProvider { root }` implements the sandbox rules
  above. Only `mlpl-repl` constructs it.
- Web/WASM local: no ambient filesystem, ever. Ships with a
  precise error ("include needs a source provider; the browser
  session has none -- run under mlpl-repl --source-dir") until an
  in-memory registry provider is worth wiring to a UI.
- Connect mode / mlpl-serve: same precise error initially (the
  server receives program STRINGS; a provider fed by uploaded
  sources is a later, deliberate feature).
- Interactive terminal REPL lines: same precise error --
  include is a script-mode construct.

## Crate ownership

- `mlpl-parser`: the `Include { path, span }` top-level AST node
  and its restrictions -- syntax only.
- NEW `mlpl-source-loader` (components/syntax-parser workspace,
  crate 3 of 5): provider trait, resolution + sandbox, the
  load-once set, cycle stack, `SourceTable`, chunk expansion.
  Pure with respect to the filesystem -- `FsProvider` lives
  behind the trait, unit-testable with an in-memory provider.
- `mlpl-repl`: `--source-dir`, `FsProvider`, chunked evaluation
  in script mode, `path:line:column` rendering.
- Evaluator: untouched.

Note: script mode's `#`-comment preprocessing runs per file,
before parsing, exactly as today.

## Test plan (maps upstream's nine required tests)

1. Parser: accepts top-level `include "x.mlpl"`; rejects
   nested/blocked placement and non-literal arguments (in
   mlpl-parser tests).
2-4. Loader (in-memory provider): cross-file definition use;
   nested include resolving relative to the INCLUDING file;
   duplicate include loads once.
5. Sandbox: missing file, absolute path, `..` escape -- each a
   structured error naming path + root (loader tests + a repl
   subprocess test).
6. Cycles: direct and indirect, full chain in the message.
7. Diagnostics: a parse error and an eval error inside an
   included file name that file and the correct line (repl
   subprocess tests).
8. Exit semantics: final `Ok`/`Err` behavior of a script with
   includes matches the same program hand-spliced.
9. Web parity: local-mode include returns the precise
   unsupported-provider error (wasm test).
Acceptance: mlplunit's `scripts/verify-native-include` passes
without host `--include`, and `check-capabilities` flips
`native-static-include` to AVAILABLE.

## Saga steps (replacing the single include-impl step)

- include-parser -- the AST node + placement/literal rules, TDD.
- include-loader -- mlpl-source-loader with in-memory-provider
  tests (expansion order, load-once, cycles, sandbox).
- include-repl -- --source-dir + FsProvider + chunked script
  evaluation + diagnostics; upstream verify-native-include run.
- include-surfaces -- web/serve/interactive precise errors,
  docs (usage, lang-reference, glossary), capability re-check.

## Open questions for review

1. Contextual `include` (proposed) vs a reserved keyword?
   Contextual costs nothing today; a keyword would break any
   existing `include` variable (none in shipped demos, but user
   workspaces exist).
2. Default source root when `--source-dir` is absent: the root
   script's directory (proposed) vs requiring the flag? The
   default makes `mlpl-repl -f tests/foo.mlpl` just work while
   staying sandboxed to the script's own tree.
3. Interactive REPL: precise error (proposed) vs allowing
   include with the cwd as root? Script-only keeps the sandbox
   story simple and matches mlplunit's use.
