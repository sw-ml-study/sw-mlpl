# Modular build: separating compilation contexts

The workspace is split into per-feature *components* (`components/<name>/`),
each its own Cargo workspace with its own `target/`. That isolates most
builds already. This doc records the finer-grained separations we want and
why -- both the ones in place and the planned ones -- so disk and wall-clock
stay bounded as the project grows.

The disk host is constrained (see CLAUDE.md "Disk-aware build hygiene"); a
single unscoped build that drags in the interpreter, the async server, the
WASM target, and proc macros can push one `target/` past 10 GB and stall.
Modularity is the structural fix, not just per-command scoping.

## Principle: a crate's heavy deps are paid by everything that builds it

Cargo compiles a crate's `[dev-dependencies]` for *every* test target in that
crate. So one heavy test (e.g. "eval every demo through the real
interpreter") forces every other test in the crate -- even a 3-line metadata
assertion -- to first rebuild the interpreter. The fix is to put the heavy
test in its own crate.

### In place: `mlpl-web-demos-smoke` (the test-crate split)

`mlpl-web-demos` holds the demo registry (generated from `demos.toml`) plus
light metadata tests (capability gating, literate links, codegen counts).
The "does every demo actually run" smoke test needs `mlpl-eval` +
`mlpl-parser` -- the whole interpreter. That smoke test lives in a separate
`mlpl-web-demos-smoke` crate, so:

- `cargo test -p mlpl-web-demos` (the light tests) compiles in ~1s.
- `cargo test -p mlpl-web-demos-smoke` (the heavy eval smoke) is a
  release-gate run, invoked deliberately, not on every metadata edit.

Before the split, a `cargo test -p mlpl-web-demos` rebuilt `mlpl-eval` and
could run for many minutes (or stall under lock contention). After, the
interpreter is only built when the smoke test is explicitly requested.

## Planned separations

These are follow-up sagas, not within-feature refactors. Each trades some
workspace-wide ergonomics for a smaller, independent `target/`.

### 1. Web UI vs. libraries

The WASM web app (`components/web*`, `components/wasm`) compiles to a
`wasm32-unknown-unknown` target tree that is several GB on its own and shares
nothing useful with native library builds. Keeping the web app's workspace
distinct from the core library workspaces means a library change never
invalidates the WASM tree and vice versa.

### 2. CUDA vs. MLX (definitely separate)

CUDA (NVIDIA/Linux) and MLX (Apple GPU) backends pull mutually exclusive,
heavy native dependency stacks (candle+CUDA toolkit vs. mlx-rs). They never
co-build on one host -- a Mac has no CUDA, a Linux box no MLX. They MUST live
in separate workspaces (`mlpl-mlx/`, a future `mlpl-cuda/`) so each host only
ever compiles the backend it can actually run, and neither stack's `target/`
bloat touches the other.

### 3. Content: demos / tutorials / paths

Demos, tutorials, and learning paths are each "data + a thin generated
const + light well-formedness tests." They already live in their own
components (`web-demos`, `web-tutorial`, `web-paths`) with content in
`.toml` and `build.rs` codegen. The separation to preserve: their light
content tests must never depend on the interpreter directly. Where a content
test needs to actually *run* MLPL (like the demo smoke), that heavy test goes
in a sibling `*-smoke` crate (as above), keeping the content crates' own test
builds instant.

### 4. Integration / regression tests to their own target

Heavy integration and regression suites (the demo smoke, `all_demos_smoke`,
end-to-end server tests) are the builds that pull the widest dependency
closures. Housing them in dedicated test-harness crates (and, longer term, a
dedicated test workspace) keeps that closure out of the day-to-day inner-loop
`target/`, so editing a library and running its unit tests never rebuilds the
integration stack. The `*-smoke` pattern above is the per-feature version of
this; a workspace-level test harness is the scaled-up version.

## Rule of thumb for new code

- A new heavy *runtime* backend -> its own workspace (never co-built with a
  rival backend).
- A test that needs the interpreter or the server -> a `*-smoke` /
  test-harness crate, not the content crate's own `tests/`.
- Content (prose, demo code, lesson text) -> `.toml` + `build.rs` codegen in
  the feature component; light well-formedness tests stay in the crate.

## Lock-file discipline across workspaces

Each component workspace's `Cargo.lock` also pins path-dependencies
owned by OTHER workspaces, so a manifest change in one component
silently strands the locks of every downstream workspace until someone
builds there. Two guards (added 2026-07-25 after a 14-workspace drift):

- `scripts/gate.sh` verifies the gated workspace's lock
  (`cargo metadata --locked`) before fmt/clippy/test.
- `scripts/check-locks.sh` sweeps every component workspace;
  `--fix` regenerates stale locks in place. Run it after changing any
  `Cargo.toml`, and commit the regenerated locks together with the
  manifest change.
