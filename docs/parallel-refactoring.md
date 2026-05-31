# Parallel Refactoring with Multiple AI Coding Agents

This document describes how to coordinate large-scale refactoring
across multiple AI coding agents (Claude, Codex,
OpenCode/GLM-5, etc.) working in parallel on separate clones of the
repository.

## Problem

The codebase has metric violations at every level of the hierarchy:

| Metric | Gate | Worst offender |
|--------|------|----------------|
| Lines per function | 25 LOC | Multiple 50+ LOC functions across eval, parser, serve |
| Functions per module | 5 | mlpl-eval modules with 10+ functions |
| Modules per crate | 5 | mlpl-eval has 76 source modules |
| Files per crate | 500 LOC/file | eval.rs (841), parser.rs (377), handlers.rs (359) |
| Crates per component | 5 | Workspace has 26 crates in a flat list |

The refactoring is additive: splitting fat crates into smaller
crates, fat modules into smaller modules, fat functions into
smaller functions. This makes it parallelizable because each agent
works on a different part of the tree and the changes do not
conflict at the file level.

## Why clones, not worktrees

Git worktrees share a single `.git` directory. For parallel
multi-agent refactoring this creates problems:

- **Lock contention.** Concurrent git operations on the same
  `.git` can block or fail.
- **Index sharing.** Agents cannot independently stage and commit
  without stepping on each other.
- **Tooling isolation.** Different agents have their own config
  directories (`.claude/`, `.codex/`, etc.) and runtime
  expectations. Clones keep these fully isolated.
- **Disk cost is low.** Each clone is roughly 50 MB for a project
  this size. Four clones cost 200 MB total.

Use one clone per agent. Each clone gets its own branch, its own
`.git`, and its own working tree.

## Dependency graph (merge order)

Refactoring branches merge bottom-up through the dependency graph.
Lower crates merge first so that upstream changes are visible when
higher crates rebase.

```
core (leaf)
  |
  +-- array (peer)       parser (peer)
  |     |                  |
  |     +------ runtime ---+
  |                |
  |     +-- eval --+
  |     |
  +-- trace
  |
  +-- viz / wasm / web-eval / web-lessons / serve
```

Merge order:

1. `mlpl-core` (rarely needs splitting)
2. `mlpl-array`, `mlpl-parser` (peers, no cross-dependency)
3. `mlpl-runtime` (depends on core, array, parser)
4. `mlpl-eval` (depends on everything -- merge last)
5. Downstream crates (`mlpl-serve`, `mlpl-viz`, `mlpl-web-*`)

## Agent assignments

Each agent owns one or two crates. No two agents touch the same
source files.

| Agent | Clone | Branch | Crate(s) | Refactoring goal |
|-------|-------|--------|----------|------------------|
| Agent A | clone-a | `refactor/eval-split` | `mlpl-eval` | Split 76-module crate into 5-6 smaller crates |
| Agent B | clone-b | `refactor/parser-runtime` | `mlpl-parser`, `mlpl-runtime` | Split fat modules, extract sub-modules |
| Agent C | clone-c | `refactor/serve-viz` | `mlpl-serve`, `mlpl-viz` | Split 300+ LOC files into sub-modules |
| Agent D | clone-d | `refactor/array-autograd` | `mlpl-array`, `mlpl-autograd` | Split fat modules, extract test helpers |

Agents are not tied to a specific AI product. Any agent slot can be
filled by Claude, Codex, OpenCode, or another tool.

## Task packet format

Each clone receives a task packet as a `TASK.md` file in its root.
The format is agent-agnostic (plain markdown, no tool-specific
config):

```markdown
# Refactoring Task: <crate-name> split

## Context
You are refactoring a Rust workspace crate. The goal is to split
an over-budget crate into smaller crates that each have <= 5
modules, <= 5 functions per module, and <= 25 LOC per function.

## Current state
- <crate> has N source modules in src/
- Largest files: <list with line counts>

## Your assignment
Split <crate> into these new crates:
1. <new-crate-1> (<which modules move here>)
2. <new-crate-2> (<which modules move here>)
...

## Rules
- Do NOT modify files outside crates/<your-crates>/ and root
  Cargo.toml
- Run `cargo test` after each move
- Every new crate gets its own Cargo.toml with minimal dependencies
- The parent crate re-exports via `pub use` for backwards
  compatibility
- Do NOT rename public API items
- Do NOT change function signatures
- Commit after each coherent move (one module or group at a time)

## Acceptance
- `cargo test` passes
- `cargo clippy --all-targets --all-features -- -D warnings` passes
- No file over 500 LOC
- No module with more than 7 functions
- No function over 50 LOC (ideally under 25)
```

## Conflict boundaries

The only shared file across agents is the root `Cargo.toml`
(workspace members list). Two strategies to manage this:

### Strategy A: integration agent owns Cargo.toml

Each refactoring agent creates new crate directories with their own
`Cargo.toml` but does NOT add them to the workspace members list.
The integration agent adds the new crates when merging each PR.

### Strategy B: designated append sections

Add section comments to the workspace members list. Each agent
appends only to their section:

```toml
[workspace]
members = [
    # --- existing ---
    "crates/mlpl-core",
    "crates/mlpl-array",
    # ...
    # --- agent-a additions (eval split) ---
    # --- agent-b additions (parser/runtime split) ---
    # --- agent-c additions (serve/viz split) ---
    # --- agent-d additions (array/autograd split) ---
]
```

Strategy A is simpler and avoids merge conflicts entirely. Strategy
B lets each agent run `cargo test` in their clone without manual
fixup but requires careful append discipline.

## Setup script

```bash
#!/bin/bash
# setup-refactor-clones.sh
#
# Creates one clone per agent, each on its own branch.

REPO="git@github.com:sw-ml-study/sw-mlpl.git"
BASE_DIR="../mlpl-refactor"

declare -A BRANCHES=(
    ["a"]="refactor/eval-split"
    ["b"]="refactor/parser-runtime"
    ["c"]="refactor/serve-viz"
    ["d"]="refactor/array-autograd"
)

mkdir -p "$BASE_DIR"

for agent in "${!BRANCHES[@]}"; do
    branch="${BRANCHES[$agent]}"
    dir="$BASE_DIR/clone-$agent"

    echo "=== Setting up clone-$agent on $branch ==="
    git clone "$REPO" "$dir"
    git -C "$dir" checkout -b "$branch" origin/main

    # Copy agent-specific task packet if it exists
    if [ -f "tasks/task-$agent.md" ]; then
        cp "tasks/task-$agent.md" "$dir/TASK.md"
    fi
done

echo "Done. Clones are in $BASE_DIR/"
```

## Integration script

Run by the integration agent (or a human) after all agents have
pushed their branches.

```bash
#!/bin/bash
# merge-refactors.sh
#
# Merges refactoring branches bottom-up in dependency order.
# Run from a clean checkout on main.

set -euo pipefail

BRANCHES=(
    "refactor/array-autograd"
    "refactor/parser-runtime"
    "refactor/serve-viz"
    "refactor/eval-split"
)

git checkout main
git pull origin main

for branch in "${BRANCHES[@]}"; do
    echo "=== Merging $branch ==="
    git fetch origin "$branch"

    # Squash merge to keep history clean
    git merge --squash "origin/$branch"

    # Fix workspace Cargo.toml if using Strategy A
    # (add new crate members that the agent created)

    # Verify
    cargo test
    cargo clippy --all-targets --all-features -- -D warnings
    cargo fmt --all -- --check

    git commit -m "refactor: merge $branch"
    echo "=== $branch merged ==="
done

echo "All branches merged. Push when ready."
```

## Handling merge conflicts

Most conflicts are trivial because agents touch different files.
The common cases:

| Conflict source | Resolution |
|-----------------|------------|
| Root `Cargo.toml` members list | Append all new members (order does not matter) |
| `Cargo.lock` | Regenerate with `cargo update --workspace` |
| A downstream crate imports a moved module | Update the import path; the parent crate's `pub use` re-export should prevent this |
| Two agents both split the same shared test helper | Keep both versions; deduplicate in a follow-up commit |

## Risk mitigation

**Biggest risk:** An agent changes a public API signature that
downstream crates depend on. Mitigate by requiring every agent to
preserve public API via re-exports from the parent crate.

**Second risk:** An agent's refactoring breaks `cargo test` in ways
that only surface when combined with another agent's changes.
Mitigate by merging bottom-up (leaf crates first) and running the
full test suite after each merge.

**Third risk:** An agent silently drops a module or function during
the move. Mitigate by requiring `cargo test` to pass in the clone
before pushing, and by running the full test suite during
integration.

## Post-merge validation

After all branches are merged, run the full quality gate:

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
sw-checklist
```

The `sw-checklist` failure and warning counts should drop
significantly. A single coordinated refactoring session splitting
`mlpl-eval` (76 modules) into 5-6 crates can retire 40+
violations at once.

## Scaling beyond four agents

The same pattern scales to more agents by partitioning at finer
granularity. Instead of assigning whole crates, assign individual
fat files:

- Agent E: split `eval.rs` (841 LOC) into sub-modules
- Agent F: split `device.rs` (443 LOC) into sub-modules
- Agent G: split `parser.rs` (377 LOC) into sub-modules

These file-level tasks are even less likely to conflict because
each agent touches exactly one file and its new sibling modules.

The constraint is merge order: file-level splits within a crate
must merge before the crate-level split that moves those files
into a new crate. Run file-level agents first, merge, then run
crate-level agents.

## Recommended refactoring targets for mlpl-eval

The largest offender. Suggested split into new crates:

| New crate | Modules to absorb | Rationale |
|-----------|--------------------|-----------|
| `mlpl-eval-grad` | `grad.rs`, `backward_*.rs` | Gradient computation is self-contained |
| `mlpl-eval-device` | `device.rs`, `env_device.rs`, `env_tensor_device.rs` | Device dispatch is a separate concern |
| `mlpl-eval-bpe` | `bpe.rs` | Tokenizer is independent of evaluation |
| `mlpl-eval-dataset` | `fetch_dataset.rs`, `image_decode.rs` | Data loading is IO-heavy, separate from pure eval |
| `mlpl-eval-model` | `model_*.rs`, `lora_*.rs` | Model construction and LoRA are a cohesive group |

The residual `mlpl-eval` crate keeps the core evaluation loop
(`eval.rs`, `env.rs`, `error.rs`) and re-exports from the new
crates for backwards compatibility.

## See also

- `docs/code_metrics.md` -- metric gates and refactoring algorithm
- `docs/loose-coupling.md` -- techniques for splitting functions
  and modules
- `COORDINATOR.md` -- single-agent coordination runbook
- `AGENTS.md` -- agent operating rules
