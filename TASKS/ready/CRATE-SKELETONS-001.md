# TASK: CRATE-SKELETONS-001

## Title
Create crate and app directory skeletons with local README and AGENTS files.

## Milestone
Saga 0 -- Foundation and contracts

## Goal
Create the top-level crate and app directories for MLPL, each with placeholder `README.md`, `AGENTS.md`, and `MILESTONE.md` files where appropriate.

## Why
Each crate is intended to be a local capsule for implementation agents. This task creates the scaffold for those capsules.

## Scope
Included:
- create crate directories
- create app directories
- add local placeholder docs

Excluded:
- creating valid Cargo manifests for every crate
- implementing code
- detailed local instructions

## Allowed directories
- `crates/`
- `apps/`

## Primary files
- `crates/mlpl-core/README.md`
- `crates/mlpl-core/AGENTS.md`
- `crates/mlpl-core/MILESTONE.md`
- similar files for all planned crates
- `apps/mlpl-repl/README.md`
- `apps/mlpl-repl/AGENTS.md`
- similar files for all planned apps

## Do not modify
- `contracts/`
- root `Cargo.toml` unless explicitly approved

## Required reading
- `COORDINATOR.md`
- `AGENTS.md`

## Contract summary
- every crate should be an isolated work capsule
- every crate should have local agent instructions
- every crate should have local progress tracking

## Acceptance tests
- manual verification of expected files
- optional script/checklist output

## Definition of done
- all planned crate directories exist
- all planned app directories exist
- placeholder local docs exist

## Escalate if
- crate naming changes are needed
- app naming changes are needed

## Suggested branch
`feat/repo/crate-skeletons-001-local-capsules`

## Suggested worktree
`../mlpl-worktrees/wt-crate-skeletons-001`
