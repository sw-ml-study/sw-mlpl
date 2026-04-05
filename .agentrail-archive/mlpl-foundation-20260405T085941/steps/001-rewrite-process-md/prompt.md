Rewrite docs/process.md to be MLPL-specific. The current file was copied from a different project and references things that do not exist here (needs-attention project, database operations, Playwright UI testing, docs/design.md, docs/learnings.md, docs/status.md).

Replace it with a short MLPL-specific process doc covering:
- Contract-first development workflow
- One task / one branch / one worktree / one PR
- Crate-local context only (agent containment)
- TDD expectations (Red/Green/Refactor)
- Pre-commit quality gates (cargo test, clippy -D warnings, cargo fmt, markdown-checker, sw-checklist)
- Escalation rules (local -> upstream contract -> coordinator summary -> selective file reveal)
- Commit message format with task IDs

Remove all references to database ops, Playwright, docs/design.md, docs/learnings.md, docs/status.md, and any other non-MLPL content. Keep the file under 200 lines.

Also mark docs/research.txt as archival by adding a note at the top: "ARCHIVAL: This is a raw conversation transcript preserved for reference. Normative decisions have been extracted into docs/*.md and docs/clarifications.txt."