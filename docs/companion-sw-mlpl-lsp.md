# Companion repo: sw-mlpl-lsp

Planning doc (forward-looking). Overview: `companion-repos.md`.

## Purpose

A Language Server Protocol implementation for MLPL, so any
LSP-speaking editor (VS Code, Neovim, Helix, and -- via a
thin shim -- the Emacs `mlpl-mode` that already ships in this
repo) gets diagnostics, completion, hover, go-to-definition,
document symbols, and test running without each editor
re-implementing them.

## Why this is a good fit for MLPL's internals

The pieces an LSP needs already exist as reusable surfaces:

- **Diagnostics** -- the parser produces file-accurate spans
  and structured errors; a `--check` parse-only mode (queued
  as an editor-tooling enabler) gives the server a cheap
  "lex + parse, report errors, do not evaluate" entry point.
- **Completion + hover** -- the builtin catalog
  (`mlpl-builtin-catalog`) is the name/signature/description
  source of truth the web `?` panel already uses; a
  machine-readable export (also queued) feeds completion items
  and hover docs directly, and `tests()` / `annotations(...)`
  cover the user-defined side.
- **Document symbols / outline** -- `def u:` definitions are
  exactly what `:fns` and imenu enumerate; the same walk
  yields LSP symbols.
- **Go-to-definition** -- references are `:u:name`, resolved by
  name against the definition site the parser recorded.
- **Test running / code lens** -- `mlpl-repl --test-events`
  emits one typed JSON event per test; the server turns those
  into per-test run results and gutter lenses (the transport
  was designed for exactly this consumer).

## Proposed scope, in order

1. Diagnostics on open/change via `--check`.
2. Document symbols (outline of `def u:` + `@test`).
3. Completion + hover from the catalog export.
4. Go-to-definition / find-references over `u:` names.
5. Test code lenses consuming `--test-events`.

## Upstream enablers (already queued)

Two small `sw-mlpl` additions unblock most of this and are
already noted in the queue as editor-tooling enablers:

- a `--check` parse-only flag (fast diagnostics with no eval);
- a machine-readable builtin-catalog export (`catalog --json`
  or equivalent), which serves both this and the web panel.

## Relationship to sw-mlpl

Consumer plus two enabler requests. It shares its needs with
`mlpl-mode.el`: whatever the LSP standardizes, the Emacs mode
can either delegate to it or mirror it, so the two should be
designed together.
