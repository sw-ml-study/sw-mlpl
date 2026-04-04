# TASK: ARRAY-CONTRACT-001

## Title
Write the initial array contract README and contract test plan.

## Milestone
Saga 0 -- Foundation and contracts

## Goal
Define the first version of the array contract for MLPL, covering shapes, ranks, indexing, reshape, and transpose at a concise contract level.

## Why
The array contract is the foundation for the tensor substrate and should be written before implementation tasks such as shape and reshape are assigned.

## Scope
Included:
- write `contracts/array-contract/README.md`
- add a small test plan file or placeholder test module list
- describe invariants and failure modes at a high level

Excluded:
- implementing the array crate
- writing full executable tests for every case
- evaluator or parser semantics

## Allowed directories
- `contracts/array-contract/`

## Primary files
- `contracts/array-contract/README.md`
- `contracts/array-contract/tests/`

## Do not modify
- `crates/mlpl-array/`
- parser/eval/runtime contracts

## Required reading
- `COORDINATOR.md`
- `AGENTS.md`

## Contract summary
- shape is first-class
- rank is derived from shape length
- indexing must be bounds-checked
- reshape preserves element order and element count
- transpose must define axis behavior clearly

## Acceptance tests
- contract review by coordinator
- contract text must be sufficient to drive subsequent array tasks

## Suggested implementation notes
- keep README concise
- use headings: Purpose, Concepts, Invariants, Operations, Errors, Open Questions

## Definition of done
- initial array contract README exists
- contract scope is clear enough for `ARRAY-SHAPE-001` and `ARRAY-RESHAPE-002`
- open questions are explicitly listed

## Escalate if
- element type model needs early commitment
- views/strides decisions block the contract

## Suggested branch
`docs/array/array-contract-001-initial-contract`

## Suggested worktree
`../mlpl-worktrees/wt-array-contract-001`
