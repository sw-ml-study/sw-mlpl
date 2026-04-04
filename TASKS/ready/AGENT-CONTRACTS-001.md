# TASK: AGENT-CONTRACTS-001

## Title
Create initial contract directory skeletons and placeholders.

## Milestone
Saga -1 — Repo compartmentalization scaffolding

## Goal
Create the initial contract directories and placeholder files for core, array, parser, runtime, eval, trace, viz, and ml contracts.

## Why
Contracts are the primary mechanism for low-context agent work. This task creates the scaffolding for contract-first development.

## Scope
Included:
- create `contracts/*` directories
- add initial `README.md` placeholders
- add `tests/` directories for each contract area

Excluded:
- detailed contract prose
- real executable contract tests

## Allowed directories
- `contracts/`

## Primary files
- `contracts/core-contract/README.md`
- `contracts/array-contract/README.md`
- `contracts/parser-contract/README.md`
- `contracts/runtime-contract/README.md`
- `contracts/eval-contract/README.md`
- `contracts/trace-contract/README.md`
- `contracts/viz-contract/README.md`
- `contracts/ml-contract/README.md`

## Do not modify
- `crates/`
- `apps/`
- root `Cargo.toml`

## Required reading
- `COORDINATOR.md`
- `AGENTS.md`

## Contract summary
- each implementation area should have a matching contract area
- each contract area should be independently readable
- contract directories should support future executable tests

## Acceptance tests
- manual verification of directory structure
- placeholder file existence

## Suggested implementation notes
- keep placeholders short
- include a one-sentence purpose and a TODO list per contract

## Definition of done
- all contract directories exist
- each has a placeholder README
- each has a `tests/` subdirectory

## Escalate if
- additional contract areas seem necessary
- naming needs adjustment to match future crate layout

## Suggested branch
`feat/repo/agent-contracts-001-contract-dirs`

## Suggested worktree
`../mlpl-worktrees/wt-agent-contracts-001`
