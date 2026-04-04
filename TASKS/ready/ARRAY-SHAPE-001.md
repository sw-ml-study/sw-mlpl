# TASK: ARRAY-SHAPE-001

## Title
Implement Shape type and shape invariants in `mlpl-array`.

## Milestone
Saga 1 — Dense tensor substrate v1

## Goal
Add a first `Shape` representation to `mlpl-array` with rank calculation, dimension access, element count calculation, and basic validation.

## Why
Shape is the most foundational concept in the dense array substrate and should be implemented before indexing, reshape, and transpose.

## Scope
Included:
- shape type
- rank computation
- dimension accessors
- total element count
- basic validation rules
- unit tests

Excluded:
- full array storage
- indexing
- reshape
- transpose
- parser/evaluator integration

## Allowed directories
- `contracts/array-contract/`
- `crates/mlpl-array/`

## Primary files
- `contracts/array-contract/README.md`
- `crates/mlpl-array/AGENTS.md`
- `crates/mlpl-array/src/lib.rs`
- `crates/mlpl-array/src/shape.rs`
- `crates/mlpl-array/src/error.rs`
- `crates/mlpl-array/tests/`

## Do not modify
- `crates/mlpl-parser/`
- `crates/mlpl-eval/`
- `apps/`
- root `Cargo.toml`

## Required reading
- `crates/mlpl-array/AGENTS.md`
- `contracts/array-contract/README.md`

## Contract summary
- shape is an ordered list of dimensions
- rank equals number of dimensions
- total element count is the product of dimensions
- invalid shape conditions must be reported explicitly

## Acceptance tests
- local `mlpl-array` unit tests
- contract-aligned shape tests

## Suggested implementation notes
- keep the type simple
- prefer explicit methods over abstractions
- use clear error types

## Definition of done
- shape type exists
- rank and element count work
- tests pass
- no out-of-scope files modified

## Escalate if
- shape validation requires a new shared error in `mlpl-core`
- contract ambiguities are discovered

## Suggested branch
`feat/array/array-shape-001-shape-type`

## Suggested worktree
`../mlpl-worktrees/wt-array-shape-001`
