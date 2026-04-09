# sw-mlpl -- Claude Instructions

## Project Overview

MLPL (Machine Learning Programming Language) is a Rust-first array and
tensor programming language platform for machine learning, visualization,
and experimentation. Inspired by APL, APL2, J, and BQN.

See `docs/` for detailed documentation:
- `docs/prd.md` -- requirements
- `docs/architecture.md` -- system architecture
- `docs/saga.md` -- implementation saga overview
- `docs/research.txt` -- design research (archival, not normative)
- `docs/clarifications.txt` -- design decisions and clarifications

## Build / Test

```bash
cargo test                                          # run all tests
cargo test -p mlpl-array                            # test one crate
cargo clippy --all-targets --all-features -- -D warnings  # lint
cargo fmt --all                                     # format
cargo fmt --all -- --check                          # check format
markdown-checker -f "**/*.md"                       # validate markdown
sw-checklist                                        # project standards
```

## CRITICAL: AgentRail Session Protocol (MUST follow exactly)

### 1. START (do this FIRST, before anything else)
```bash
agentrail next
```
Read the output carefully. It contains your current step, prompt,
plan context, and any relevant skills/trajectories.

### 2. BEGIN (immediately after reading the next output)
```bash
agentrail begin
```

### 3. WORK (do what the step prompt says)
Do NOT ask "want me to proceed?". The step prompt IS your instruction.
Execute it directly.

**Every step MUST follow TDD (Red/Green/Refactor):**
- RED: Write failing tests first that define expected behavior
- GREEN: Write minimal code to make tests pass
- REFACTOR: Clean up while keeping tests green

### 4. COMMIT (after the work is done)
Commit your code changes with git. Use `/mw-cp` for the checkpoint
process (pre-commit checks, docs, detailed commit, push).

**The `/mw-cp` checkpoint process includes:**
1. `cargo test` -- ALL tests must pass, no exceptions
2. `cargo clippy --all-targets --all-features -- -D warnings` -- ZERO warnings
3. `cargo fmt --all` -- all code formatted
4. `cargo fmt --all -- --check` -- verify no formatting changes remain
5. `markdown-checker -f "**/*.md"` -- ASCII-only markdown (if docs changed)
6. `sw-checklist` -- project standards check
7. Update docs if any code behavior changed
8. Detailed commit message with task context
9. `git push`

**Never skip quality gates. Never suppress warnings. Fix issues, do not defer them.**

### 5. COMPLETE (LAST thing, after committing)
```bash
agentrail complete --summary "what you accomplished" \
  --reward 1 \
  --actions "tools and approach used" \
  --next-slug "next-step-slug" \
  --next-prompt "what the next step should do" \
  --next-task-type "task-type"
```
- If the step failed: `--reward -1 --failure-mode "what went wrong"`
- If the saga is finished: add `--done`

### 6. STOP (after complete, DO NOT continue working)
Do NOT make any further code changes after running agentrail complete.
Any changes after complete are untracked and invisible to the next session.
If you see more work to do, it belongs in the NEXT step, not this session.

## Live Demo (GitHub Pages) Deploy

The `.github/workflows/pages.yml` workflow ONLY uploads the committed
`./pages/` directory -- it does NOT build from source. Source-only
commits to `apps/mlpl-web/` will never reach the live demo at
<https://sw-ml-study.github.io/sw-mlpl/>.

Whenever you change `apps/mlpl-web/` (Rust, `index.html`, CSS, assets,
or the `demos.rs` list), you MUST rebuild and commit `pages/`:

```bash
./scripts/build-pages.sh    # trunk build --release into pages/
git add pages/
git commit -m "chore(pages): rebuild for <what changed>"
git push
```

Then the push-triggered Pages workflow deploys the new `pages/`.
Verify with `gh run list --workflow=pages.yml --limit 1`.

## NEVER run sw-install without an explicit request

The user keeps a stable installed `mlpl-repl` (and other mlpl
binaries) under `~/.local/softwarewrighter/bin/` for use OUTSIDE
this dev session. Reinstalling between feature commits would
overwrite that stable binary with an in-progress build and break
the user's other work.

- Do NOT run `sw-install` after a feature commit, a release
  commit, or a quality-gate green-light.
- Do NOT run `sw-install` as part of `/mw-cp` or any "ship it"
  flow.
- Only run `sw-install` when the user explicitly asks for it
  ("install the binaries", "sw-install", "update my installed
  mlpl", etc.).
- `cargo build --release` is fine on its own -- it stays inside
  `target/release/` and does not touch the installed binaries.

If you are unsure whether the user wants an install, ASK first.

## Key Rules

- **Do NOT skip steps** -- the next session depends on accurate tracking
- **Do NOT ask for permission** -- the step prompt is the instruction
- **Do NOT continue working** after `agentrail complete`
- **Commit before complete** -- always commit first, then record completion
- **TDD is mandatory** -- write tests before implementation code
- **Quality gates are mandatory** -- every commit passes all checks
- **Push after every commit** -- enables testing on other systems

## Useful Commands

```bash
agentrail status          # Current saga state
agentrail history         # All completed steps
agentrail plan            # View the plan
agentrail next            # Current step + context
```

## Architecture

Cellular monorepo with narrow crates and matching contract directories.

Dependency flow:
`core -> array/parser -> runtime -> eval -> trace -> viz/wasm/apps -> ml`

- Parser and array are peers (parser does NOT depend on array)
- Each crate owns its own error types (not centralized in core)
- Contracts are prose specs for now; executable tests go in crate-local tests/

### Key directories

- `contracts/` -- compact local contracts for each implementation area
- `crates/` -- library capsules
- `apps/` -- user-facing entry points (repl, web, lab)
- `TASKS/` -- task packets for agents
- `docs/` -- architecture, PRD, and saga documentation

## Agent Coordination

- One task / one branch / one worktree / one PR
- Agents start with minimal context (local crate + local contract)
- See `COORDINATOR.md` and `AGENTS.md` for full rules
- Escalate before touching files outside allowed directories

## Key Design Decisions

- TDD with Red/Green/Refactor cycle
- ASCII-first syntax, Unicode later
- MVP is CLI + JSON trace export (no Yew/WASM viewer in MVP)
- Syntax design spike required before parser implementation
- `mlpl-core` stays small: spans, identifiers, small shared types only
- research.txt is archival reference only; normative docs are in docs/*.md
