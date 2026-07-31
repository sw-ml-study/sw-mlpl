ChatGPT














I have a large github repo with many "components" that are each mult-create Rust workspaces.  When building the repo's main app, there are a lof of compiles producing a lot of build artifacts across many ./target dirs.  cargo clean in one causes ripple effects for others.  What is a strategy for capturing enough of the dependency binary artifacts to build other components while reducing total disk space usage?

I'll frame this as a repo/build-cache architecture problem: how Cargo chooses artifact locations, what can be safely shared, and which cleanup policies avoid breaking downstream component builds.

Thought for 5s
You want less disk usage without forcing expensive rebuilds across sibling Rust workspaces.

Recommended strategy
Use three tiers, not repeated cargo clean.

1. Centralize build output by component class
Instead of each component using ./target, set CARGO_TARGET_DIR to a controlled shared area:

export CARGO_TARGET_DIR=/large/build-cache/myrepo/main-target
Cargo officially supports changing the target directory via CARGO_TARGET_DIR, build.target-dir, or --target-dir. 

But do not use one global target dir for everything. Split by compatibility boundary:

.build/
  target/
    stable-x86_64-debug/
    stable-x86_64-release/
    nightly-x86_64-debug/
    wasm32-debug/
    embedded-thumbv7em/
Key dimensions:

Dimension	Why it matters
toolchain	stable/nightly/rust version changes invalidate lots
target triple	host vs wasm vs embedded artifacts differ
profile	debug/release/test/bench differ
feature set	major feature combos may duplicate artifacts
component family	prevents one app's cleanup from nuking unrelated builds
Avoid a single shared target dir if people will run cargo clean, because cargo clean without options deletes the whole target directory. 

2. Add sccache as the rebuild safety net
Use sccache for compiled dependency reuse:

export RUSTC_WRAPPER=sccache
export SCCACHE_DIR=/large/build-cache/sccache
sccache wraps the compiler and reuses cached compilation outputs when the same compile happens again. 

This lets you be more aggressive about pruning old target directories, because many dependency crates can come back from cache instead of full recompilation.

3. Replace cargo clean with a repo-aware pruning tool
Do not run plain cargo clean except for emergencies.

Instead, write a small Rust tool/script that:

Finds all target dirs or centralized target dirs.

Computes size and last-access/modified time.

Keeps:

artifacts for the main app,

artifacts for active components,

the newest debug, release, and test outputs,

high-cost dependency artifacts.

Deletes:

old incremental dirs,

stale test/bench artifacts,

old target triples,

old profile directories,

stale component-local ./target dirs.

Good first cleanup targets:

target/debug/incremental/
target/release/incremental/
target/tmp/
target/.rustc_info.json
old component ./target dirs
stale target/<triple>/debug/incremental
The biggest win is usually deleting incremental artifacts before deleting compiled dependencies.

Practical layout I'd use
At repo root:

repo/
  .cargo/config.toml
  tools/
    cargo-prune-targets/
  components/
    comp-a/
    comp-b/
    comp-c/
  .build/
    target/
    sccache/
.cargo/config.toml:

[build]
target-dir = ".build/target/default"

[env]
SCCACHE_DIR = { value = ".build/sccache", relative = true }
RUSTC_WRAPPER = "sccache"
For special builds:

CARGO_TARGET_DIR=.build/target/wasm32-debug \
cargo build --target wasm32-unknown-unknown
For main app builds:

CARGO_TARGET_DIR=.build/target/main-debug \
cargo build -p main-app
Cleanup policy
Use a command like:

cargo xtask prune-targets --max-gb 80 --keep-days 14 --keep-main
Policy:

Artifact class	Keep?
main app current profile	yes
common dependency artifacts	yes
current component artifacts	yes
incremental build dirs older than a few days	no
old test/bench artifacts	usually no
old target triples	no unless active
old release artifacts	keep only selected
component-local ./target dirs	migrate/delete
Important warning
A shared CARGO_TARGET_DIR reduces duplication, but makes cargo clean more dangerous. Cargo's own clean command removes generated artifacts from the chosen target directory, and with no options it removes the entire target directory. 

So the rule should be:

No raw cargo clean in this repo.
Use repo-clean / xtask prune instead.
Best architecture
Use centralized target dirs for active artifacts, sccache for reusable compiler outputs, and a repo-aware pruning tool instead of cargo clean.

That gives you:

lower duplicate disk usage,

fewer ripple-effect rebuilds,

predictable cleanup,

easy reporting of "what is using disk,"

and a path toward CI/local cache sharing later.



