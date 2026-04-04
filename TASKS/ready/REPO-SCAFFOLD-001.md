# TASK: REPO-SCAFFOLD-001

## Title
Create the initial cellular monorepo directory structure.

## Milestone
Saga -1 — Repo compartmentalization scaffolding

## Goal
Create the top-level repository structure for MLPL as a cellular monorepo, including the main directory layout for docs, tasks, contracts, crates, apps, demos, examples, scripts, and tools.

## Why
This task creates the physical structure that allows later work to be compartmentalized by crate and contract. It is the first step toward low-context agent development.

## Scope
Included:
- create top-level directories
- create empty placeholder directories for contracts, crates, and apps
- create placeholder markdown files where required

Excluded:
- writing all docs in detail
- implementing Rust code
- populating all crate internals

## Allowed directories
- `./`

## Primary files
- `README.md`
- `AGENTS.md`
- `COORDINATOR.md`
- `WORKTREES.md`
- `TASKS/`
- `docs/`
- `contracts/`
- `crates/`
- `apps/`

## Do not modify
- files outside the repository
- CI/provider configuration unless explicitly needed

## Required reading
- `COORDINATOR.md`
- `AGENTS.md`

## Contract summary
- repository must support task-based compartmentalized work
- repository must expose contracts as first-class directories
- repository must support future crate-local agent instructions

## Acceptance tests
- visual/manual verification of directory tree
- optional script or checklist output proving required paths exist

## Suggested implementation notes
- use `.gitkeep` where needed for empty directories
- keep placeholder files short and neutral

## Definition of done
- required directory tree exists
- placeholder files exist
- no Rust implementation added yet
- repository layout matches docs

## Escalate if
- there is disagreement about top-level directory naming
- a root file needs content that belongs to another task

## Suggested branch
`feat/repo/repo-scaffold-001-cellular-layout`

## Suggested worktree
`../mlpl-worktrees/wt-repo-scaffold-001`
