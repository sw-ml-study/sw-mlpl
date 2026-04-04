# mlpl-array agent notes

## Purpose

`mlpl-array` defines dense array/tensor structures, shapes, indexing, layout, reshape, transpose, and related structural operations.

## This crate should know
- shapes
- ranks
- dimensions
- storage layout
- indexing
- structural transforms

## This crate should not know
- parser syntax
- evaluator semantics
- trace serialization
- Yew UI
- ML backend adapters

## Allowed dependencies
- `mlpl-core`

## Module style
- keep modules small
- prefer pure functions
- keep error cases explicit
- do not introduce broad trait abstractions unless required

## Testing

Run:
- `cargo test -p mlpl-array`
- contract tests for array behavior

## Escalate if
- shape or error contracts require changes in `mlpl-core`
- an API change would affect `mlpl-runtime`
