# TASK: PARSER-LEX-001

## Title
Implement the first lexer for MLPL numeric literals and core punctuation.

## Milestone
Saga 1 / early Saga 2 bridge — Parser foundation

## Goal
Create the first lexer in `mlpl-parser` that can tokenize numeric literals, identifiers if needed, and the core punctuation required for simple expressions and array literals.

## Why
A small lexer can be developed in parallel with early array work and will unblock the first parser/evaluator milestones.

## Scope
Included:
- token definitions
- numeric literal lexing
- punctuation lexing for simple expressions
- source span tracking
- lexer tests

Excluded:
- full parser
- evaluator
- array semantics
- Unicode syntax

## Allowed directories
- `contracts/parser-contract/`
- `crates/mlpl-parser/`
- optionally `crates/mlpl-core/` only if spans are defined there and task allows read-only use

## Primary files
- `contracts/parser-contract/README.md`
- `crates/mlpl-parser/AGENTS.md`
- `crates/mlpl-parser/src/lib.rs`
- `crates/mlpl-parser/src/token.rs`
- `crates/mlpl-parser/src/lexer.rs`
- `crates/mlpl-parser/tests/`

## Do not modify
- `crates/mlpl-eval/`
- `crates/mlpl-array/`
- `apps/`
- root `Cargo.toml`

## Required reading
- `crates/mlpl-parser/AGENTS.md`
- `contracts/parser-contract/README.md`

## Contract summary
- lexer must produce stable token kinds
- tokens must include source spans
- invalid input must produce explicit lexer errors
- implementation is ASCII-first

## Acceptance tests
- local `mlpl-parser` unit tests
- contract-aligned lexer tests

## Suggested implementation notes
- keep token set minimal
- avoid parser assumptions leaking into lexer design
- make spans explicit from day one

## Definition of done
- lexer tokenizes numeric literals and core punctuation
- tests pass
- output is stable enough for the next parser task

## Escalate if
- span type in `mlpl-core` does not yet exist and is needed
- parser contract is not yet sufficient

## Suggested branch
`feat/parser/parser-lex-001-basic-lexer`

## Suggested worktree
`../mlpl-worktrees/wt-parser-lex-001`
