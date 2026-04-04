# TASK: WORKSPACE-BOOT-001

## Title
Create initial Cargo workspace and minimal compile/test smoke path.

## Milestone
Saga 0 -- Foundation and contracts

## Goal
Create the root Cargo workspace for `sw-mlpl` with a minimal set of placeholder crates sufficient to run `cargo check` and basic smoke tests.

## Why
This creates the first executable foundation for the repository and allows later crate work to happen inside a real Rust workspace.

## Scope
Included:
- create root `Cargo.toml`
- create minimal `Cargo.toml` and `src/lib.rs` or `src/main.rs` for initial crates/apps
- ensure `cargo check` succeeds

Excluded:
- meaningful semantics
- detailed tests for language behavior
- broad dependency setup

## Allowed directories
- `Cargo.toml`
- `crates/`
- `apps/`

## Primary files
- `Cargo.toml`
- `crates/mlpl-core/Cargo.toml`
- `crates/mlpl-core/src/lib.rs`
- `crates/mlpl-array/Cargo.toml`
- `crates/mlpl-array/src/lib.rs`
- `crates/mlpl-cli/Cargo.toml`
- `crates/mlpl-cli/src/lib.rs`
- `apps/mlpl-repl/Cargo.toml`
- `apps/mlpl-repl/src/main.rs`

## Do not modify
- detailed contracts
- milestone docs outside local updates

## Required reading
- `COORDINATOR.md`
- root `AGENTS.md`

## Contract summary
- workspace should build successfully in minimal form
- initial crates should exist as independent units
- no broad implementation should be added in this task

## Acceptance tests
- `cargo check`
- `cargo test` for smoke-only cases if included

## Suggested implementation notes
- start with a very small set of workspace members
- add more members later if that reduces friction

## Definition of done
- root workspace exists
- minimal workspace builds
- placeholder Rust source compiles

## Escalate if
- workspace dependency structure becomes unclear
- additional root tooling files are needed

## Suggested branch
`feat/repo/workspace-boot-001-minimal-workspace`

## Suggested worktree
`../mlpl-worktrees/wt-workspace-boot-001`
