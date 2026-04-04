# MLPL Worktree Registry

This file tracks active and recent task worktrees for the `sw-mlpl` repository.

## Status Legend

- `ready` — task prepared, no worktree yet
- `active` — worktree exists, task in progress
- `pr-open` — PR opened, awaiting review/integration
- `merged` — merged to `main`
- `blocked` — task blocked
- `abandoned` — task intentionally stopped

---

## Active and Recent Worktrees

| Task ID | Area | Branch | Worktree Path | Status | Owner | PR | Notes |
|---|---|---|---|---|---|---|---|
| REPO-SCAFFOLD-001 | repo | feat/repo/repo-scaffold-001-cellular-layout | `../mlpl-worktrees/wt-repo-scaffold-001` | ready | unassigned |  | initial repo layout |
| AGENT-CONTRACTS-001 | repo | feat/repo/agent-contracts-001-contract-dirs | `../mlpl-worktrees/wt-agent-contracts-001` | ready | unassigned |  | create contracts skeleton |
| ARRAY-SHAPE-001 | array | feat/array/array-shape-001-shape-type | `../mlpl-worktrees/wt-array-shape-001` | ready | unassigned |  | first array substrate task |
| PARSER-LEX-001 | parser | feat/parser/parser-lex-001-basic-lexer | `../mlpl-worktrees/wt-parser-lex-001` | ready | unassigned |  | first parser task |

---

## Usage Notes

- Every task should have at most one active worktree.
- Use one worktree per task branch.
- Delete worktrees after merge.
- Keep this file updated as part of coordinator/integration workflow.
