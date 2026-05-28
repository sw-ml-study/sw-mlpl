# mlpl-parser agent notes

## Purpose

`mlpl-parser` defines tokens, lexing, parsing, AST, and syntax diagnostics for MLPL.

## This crate should know
- token kinds
- source spans
- syntax structure
- parse diagnostics

## This crate should not know
- evaluator internals
- tensor semantics beyond syntax needs
- trace serialization
- UI concerns

## Allowed dependencies
- `mlpl-core`

## Testing

Run:
- `cargo test -p mlpl-parser`

## Escalate if
- span types need changes in `mlpl-core`
- parser contract becomes ambiguous
