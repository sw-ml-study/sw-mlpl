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

## Disk-aware build hygiene (Saga 21+)

The Mac dev host is disk-constrained; full unscoped
`cargo` runs across this workspace can push `target/`
to 30+ GB and fill the disk mid-session. Saga 21 added
a heavy HTTP stack (axum + hyper + tokio + reqwest +
rustls) that compounds the problem -- and the dev host
move to Linux is the longer-term fix, but until then
follow these rules:

### Build only what you need

- **Default to scoped commands.** `cargo build -p <crate>`
  / `cargo test -p <crate>` / `cargo test -p <crate>
  --test <file>` instead of workspace-wide runs when
  you are working on a single crate.
- **Skip `cargo bench` and `./scripts/build-pages.sh`
  unless explicitly required.** `bench` pulls Criterion
  + plotters; `build-pages.sh` builds the WASM target
  tree (separate `target/wasm32-unknown-unknown/`,
  several GB on its own). Only run pages/ rebuild when
  `apps/mlpl-web/` actually changed (the live demo gate
  in CLAUDE.md still applies, but only when the web
  source changed).
- **Selective clean** is OK: `cargo clean -p <crate>`
  removes only that crate's artifacts.

### Clean policy

- **Run `cargo clean` if `target/` exceeds ~10 GB.**
  Check with `du -sh target/`. The full
  release-build-only baseline for this workspace is
  ~500 MB; a complete /mw-cp pass adds ~3-4 GB more
  (clippy + test profile). Anything past 10 GB is
  almost always accumulated incremental cruft from
  prior sessions, not load-bearing state.
- **Always `cargo clean --release` after a release
  step** -- release artifacts aren't needed once
  shipped. Saves ~12-15 GB on a fully-built tree.
- **Measure free space + target/ before and after**
  any clean so the user can see what was recovered.

### /mw-cp on a constrained disk

The full /mw-cp gate is still required for code-
affecting commits, but if the commit is docs-only or a
version-bump-only release step (e.g., the final step
of a saga), prefer scoped tests for the changed crates
plus full clippy / fmt / markdown / sw-checklist.
Document the scoping rationale in the commit message.

### Pre-save binaries before destructive cleans

When the user explicitly authorizes a `cargo clean` to
free space mid-session, copy any in-flight binaries
the session needs into a safe scratch dir BEFORE the
clean -- e.g., `cp target/release/mlpl-serve
/tmp/mlpl-binaries/` -- so the binary survives the
clean and can still be exercised by manual smoke
tests. Do NOT copy into
`~/.local/softwarewrighter/bin/` -- that is the user's
stable-install path (see "NEVER run sw-install"
below).

### Future partition saga

The workspace currently bundles five compilation
contexts in one `target/`: native interpreter, async
server, WASM target, proc macros, benchmarks.
Splitting into separate Cargo workspaces (e.g., a
`mlpl-web/` workspace, a `mlpl-serve/` workspace, an
`mlpl-mlx/` workspace, a future `mlpl-cuda/`
workspace) would each get its own `target/` --
trading workspace-wide ergonomics for disk
discipline. Treat that as a planned follow-up saga,
not a within-current-saga refactor.

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
