# MLPL Development Process

## Overview

This document describes the development workflow for MLPL. The project
uses contract-first development, strict TDD, and agent containment
to build a cellular monorepo of narrow Rust crates.

## Core Workflow

### Contract-First Development

Every crate has a matching contract directory under `contracts/`.
Contracts are prose specs that define behavior before code exists.

1. Write or review the contract for the target crate
2. Write tests that encode the contract's requirements (RED)
3. Implement minimal code to pass tests (GREEN)
4. Refactor while tests stay green (REFACTOR)

### TDD: Red / Green / Refactor

All code changes follow strict TDD:

- **RED**: Write a failing test that defines expected behavior.
  Confirm it fails for the right reason.
- **GREEN**: Write the simplest code that makes the test pass.
  Don't optimize yet.
- **REFACTOR**: Clean up while keeping tests green. Remove
  duplication, improve names, simplify logic.
- **REPEAT**: Next piece of functionality.

### One Task / One Branch / One Worktree / One PR

Each unit of work gets its own branch, worktree, and PR.
No mixing unrelated changes. This keeps diffs reviewable
and rollbacks clean.

## Agent Containment

Agents operate with crate-local context only:

- **Allowed**: Local crate source + local contract README
- **Escalation path** (when more context is needed):
  1. Read the local contract
  2. Read the upstream contract (dependency crate)
  3. Read COORDINATOR.md summary
  4. Request selective file reveal from coordinator

Never touch files outside your assigned crate without escalation.
See `COORDINATOR.md` and `AGENTS.md` for full rules.

## Pre-Commit Quality Gates

**MANDATORY before every commit. No exceptions.**

Run these in order. All must pass.

### 1. Tests

```bash
cargo test
```

All tests must pass. Do not disable or skip failing tests.

### 2. Lint

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

Zero warnings. Never suppress with `#[allow(...)]`. Fix properly.

### 3. Format

```bash
cargo fmt --all
cargo fmt --all -- --check
```

All code formatted. Verify no changes remain.

### 4. Markdown (if docs changed)

```bash
markdown-checker -f "**/*.md"
```

ASCII-only markdown. Use `--fix` for auto-fixable issues.

### 5. Project Standards

```bash
sw-checklist
```

All checklist items must pass.

### 6. Documentation

Update relevant docs if behavior changed:

- `docs/architecture.md` if system design changed
- `docs/prd.md` if requirements changed
- Contract READMEs if crate behavior changed
- `CLAUDE.md` if development patterns changed

## Commit Messages

Format:

```
type(scope): Short summary

Detailed explanation of what and why.

Task: TASK-ID (if applicable)

Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Scope is the crate name when applicable (e.g., `feat(mlpl-array): ...`).

## Push Policy

Push immediately after every commit. This enables testing on other
systems and provides backup.

## Quality Standards

- Zero clippy warnings (enforced with `-D warnings`)
- All code formatted with `cargo fmt`
- Rust 2024 edition idioms
- Inline format arguments: `format!("{name}")` not `format!("{}", name)`
- Inner doc comments for modules: `//!`
- Files under 500 lines (prefer 200-300)
- Functions under 50 lines (prefer 10-30)
- Maximum 3 TODO comments per file
- Never commit FIXMEs (fix immediately)

## Testing Strategy

### Unit Tests

In-file `#[cfg(test)]` modules. Test pure logic and edge cases.

### Integration Tests

In `tests/` directories within each crate. Test cross-module behavior.

### What to Test

- Contract requirements (primary driver)
- Edge cases (empty input, max values, error conditions)
- Regressions (add test for every bug fix)

## Build

```bash
cargo test                  # run all tests
cargo test -p mlpl-array    # test one crate
cargo check --workspace     # type-check everything
cargo build                 # full build
```

## Key References

- `docs/prd.md` -- product requirements
- `docs/architecture.md` -- system architecture
- `docs/saga.md` -- implementation saga
- `docs/clarifications.txt` -- design decisions
- `contracts/` -- per-crate behavioral contracts
- `COORDINATOR.md` -- agent coordination rules
- `AGENTS.md` -- agent containment rules
