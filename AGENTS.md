# MLPL Agent Rules

## Mission

MLPL is a Rust-first tensor and array programming language with visual debugging and ML experimentation support.

## Development model

Work is task-based and contract-first.

Default rule:

- one task
- one branch
- one worktree
- one PR

## Context containment

Agents should begin with only:

- their task packet
- their local crate directory
- their contract directory
- their local crate `AGENTS.md`

Do not browse the wider repo unless required by escalation.

## Allowed read scope

1. local crate
2. local contract
3. direct upstream public API, only if needed
4. coordinator-provided summary
5. selectively revealed external files

## Forbidden behavior

- do not refactor unrelated crates
- do not edit root workspace files unless task explicitly allows it
- do not widen public APIs casually
- do not browse apps, UI, or docs for implementation inspiration unless the task requires it
- do not solve adjacent problems “while here”

## Dependency rule

Prefer one-way dependencies:

`core → array/parser → runtime → eval → trace → viz/wasm/apps → ml`

## PR rule

Open a small PR when local acceptance tests pass.
Use task ID in branch, PR title, and final squash commit.

## Escalation

Escalate when:

- the contract is ambiguous
- a new upstream API is needed
- a forbidden file would need modification
- local implementation reveals an architecture conflict
