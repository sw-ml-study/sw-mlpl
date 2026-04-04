# Repo Structure

This repository is named `sw-mlpl`. The language and platform are called **MLPL**.

## Top-level layout

- `contracts/` — compact, local contracts for each implementation area
- `crates/` — library capsules
- `apps/` — user-facing entry points
- `TASKS/` — task packets for agents
- `docs/` — architecture, PRD, and saga documentation

## Capsule rule

Each crate should eventually contain:
- `README.md`
- `AGENTS.md`
- `MILESTONE.md`
- `src/`
- `tests/`

Each contract area should contain:
- `README.md`
- `tests/`
