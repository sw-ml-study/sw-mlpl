# MLPL Coordinator Runbook

This repository is `sw-mlpl`.
The product, language, and platform described in docs and README are called **MLPL**.

## Purpose

This file defines the operating procedure for coordinating multiple AI coding agents working in parallel on MLPL.

The goals are:

- low friction
- low interference
- narrow task scope
- contract-first development
- short-lived branches
- reproducible integration
- minimal agent context

---

## Core Rule

Every implementation task should follow this pattern:

- one task
- one branch
- one worktree
- one PR

Default merge style: **squash merge**.

---

## Repository Model

The repository is a **cellular monorepo**.

Each crate is a capsule with:
- its own `README.md`
- its own `AGENTS.md`
- its own tests
- narrow dependencies
- a matching contract area in `contracts/`

Agents should work locally in one capsule plus one contract directory.

---

## Naming Rules

### Repo
- Git repo name: `sw-mlpl`
- Language/platform name in docs: `MLPL`

### Task IDs
Format:
`<AREA>-<TOPIC>-<NNN>`

Examples:
- `ARRAY-SHAPE-001`
- `PARSER-LEX-001`
- `TRACE-EVENT-001`

### Branch names
Format:
`feat/<area>/<task-id-lower>-<slug>`
`fix/<area>/<task-id-lower>-<slug>`
`docs/<area>/<task-id-lower>-<slug>`
`spike/<area>/<task-id-lower>-<slug>`

Examples:
- `feat/array/array-shape-001-shape-type`
- `feat/parser/parser-lex-001-basic-lexer`

### Worktree names
Format:
`wt-<task-id-lower>`

Examples:
- `wt-array-shape-001`
- `wt-parser-lex-001`

### PR titles
Format:
`[<area>] <TASK-ID>: <summary>`

Examples:
- `[array] ARRAY-SHAPE-001: add Shape type and invariants`
- `[parser] PARSER-LEX-001: implement numeric and punctuation lexer`

---

## Branch and Worktree Procedure

### Create a task branch and worktree

From the main checkout:

```bash
git fetch origin
git checkout main
git pull --ff-only origin main

git worktree add ../mlpl-worktrees/wt-array-shape-001   -b feat/array/array-shape-001-shape-type   origin/main
```

### Agent works only in that worktree

The agent should begin in:
- the assigned worktree
- the assigned crate directory
- the assigned contract directory

The agent should not browse the wider repo unless escalation permits it.

---

## Task Lifecycle

### 1. Ready
Task file exists in `TASKS/ready/`.

### 2. Active
Task file is moved to `TASKS/active/`.
Branch and worktree are created.
Agent begins implementation.

### 3. Blocked
If blocked, move task file to `TASKS/blocked/` or keep in `active/` with a blocked note.

### 4. PR Open
PR references the task ID and contract.

### 5. Done
After merge:
- move task file to `TASKS/done/`
- update `WORKTREES.md`
- remove worktree
- delete branch

---

## Context Containment Rules

Agents start with only:
- the task packet
- the local crate `AGENTS.md`
- the local contract README/tests
- the allowed directories listed in the task

Agents should not read:
- unrelated crates
- apps
- top-level docs
- downstream consumers
unless the task explicitly allows it or escalation approves it.

---

## Escalation Ladder

### Level 0 -- local only
Allowed:
- assigned crate
- assigned contract
- local tests

### Level 1 -- upstream public API
Allowed only if needed:
- direct upstream crate public API
- direct upstream contract

### Level 2 -- coordinator summary
Coordinator provides a short summary of an external requirement.

### Level 3 -- selective file reveal
One or two specific external files are approved.

### Level 4 -- design decision
Human/coordinator resolves ambiguity or architecture conflict.

---

## Allowed Scope Rule

Each task packet must include:
- allowed directories
- prohibited directories
- acceptance tests
- definition of done
- escalation triggers

If a task would require touching files outside allowed directories, the agent must escalate before making the change.

---

## Ownership Rules

### Coordinator / integration agent owns
- root `Cargo.toml`
- `Cargo.lock`
- root `README.md`
- top-level docs
- CI config
- milestone summaries
- cross-cutting refactors

### Crate agent owns
- assigned crate files
- assigned contract files
- local tests
- local README/AGENTS/MILESTONE updates if task allows

---

## Merge Policy

Default:
- small PR
- green tests
- squash merge

The integration agent handles:
- rebase if needed
- conflict resolution
- broader test execution
- merge
- cleanup

---

## Cleanup Procedure

After merge:

```bash
git worktree remove ../mlpl-worktrees/wt-array-shape-001
git branch -d feat/array/array-shape-001-shape-type
git push origin --delete feat/array/array-shape-001-shape-type
```

Update:
- `WORKTREES.md`
- task location in `TASKS/`
- milestone docs if needed

---

## What to Optimize For

Prefer:
- narrow tasks
- local contracts
- short branch lifetime
- early integration
- explicit boundaries

Avoid:
- broad multi-crate tasks
- drive-by refactors
- long-lived branches
- direct commits to `main`
- widening public APIs casually
