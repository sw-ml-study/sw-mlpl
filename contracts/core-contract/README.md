# Core Contract

## Purpose

Define the shared low-level types used across multiple MLPL crates.
`mlpl-core` is deliberately small -- it holds only types that two or
more crates genuinely need.

## Key Types and Concepts

### Span

A byte-offset range into source text.

- `Span { start: usize, end: usize }`
- Used by parser for token locations and by eval/trace for
  correlating execution steps back to source
- `start` is inclusive, `end` is exclusive

### Identifier

A validated name string.

- Must start with an ASCII letter or underscore
- May contain ASCII letters, digits, and underscores
- Possibly interned or wrapped in a newtype for cheap cloning

## Invariants

- `Span.start <= Span.end`
- Identifiers are never empty
- Identifier validation is deterministic (same input -> same result)
- Core types must not depend on any other MLPL crate

## Error Cases

- Core does not define a shared error enum
- Each downstream crate owns its own error types
- Core may provide small utility traits for error context if needed

## What This Contract Does NOT Cover

- Parser errors, array errors, runtime errors (those are crate-local)
- AST nodes, token kinds, array shapes
- Anything that only one crate needs
