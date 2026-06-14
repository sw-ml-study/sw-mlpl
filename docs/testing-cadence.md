# Testing cadence: when to run the slow tests

The MLPL interpreter walks the AST one node at a time through a dispatch
table. In the `dev` (debug) profile those calls are not inlined, so any
test dominated by an interpreter loop -- training demos, the web-demo
smoke, ViT/LM end-to-end tests -- runs **30-60x slower than `--release`**.

That cost asymmetry sets the whole strategy: **a cheap proxy in the inner
loop; the expensive suite only at blast-radius boundaries.** Heavy tests
are a GATE, not an iteration tool.

## Decide by blast radius

Ask: "could this change break a demo I am not looking at?"

| What you touched | Risk | Run, and when |
| --- | --- | --- |
| A leaf crate (one viz renderer, a web component, one builtin's math), or docs | Low | That crate's own tests, in `dev`, on save. Nothing heavier. |
| Shared/core: `mlpl-eval` dispatch, `mlpl-runtime`, the parser, autograd/tape, the model DSL, or **any demo source string** | High | A fast targeted test in the inner loop; the heavy smoke in `--release` **before you push**. |
| Saga close / release boundary | -- | The full heavy suite once, in `--release`. |

## The inner-loop proxy

Do not run a 150-step training demo to check that it works. Write a small
test that hits the **same code path** with a handful of steps. Example:
`components/eval/crates/mlpl-eval/tests/train_val_curve_tests.rs` exercises
the `train_val_curve` dispatch, the concat loss-accumulation pattern, and
the "Watch a Model Learn" demo's exact structure in miniature (4 steps) --
it runs in ~0.01s and gives the same regression signal the full demo would.

Heavy demos live in the heavy bucket (`SKIP_DEMOS` in the web-demos
`registry.rs` test, and `HEAVY_TRAINING` in `all_demos_smoke.rs`), each
paired with a fast proxy. They are NOT in the default quick suite.

## The heavy gates (release, on demand)

```bash
# Web demo registry, heavy entries:
( cd components/web-demos && \
  cargo test -p mlpl-web-demos --release --test registry \
    every_heavy_web_demo_runs -- --ignored )

# demos/*.mlpl files, heavy entries:
( cd components/eval && \
  cargo test -p mlpl-eval --release --test all_demos_smoke -- --ignored )
```

Run these before pushing a broad change to eval/runtime/builtins/demos, and
once at every saga close.

## Two hard constraints

- **One `cargo`/`trunk` at a time on the shared `target/`.** Two concurrent
  builds serialize on cargo's build lock and *look* like a hang. Chain
  commands with `&&` (one shell, sequential) or use `scripts/serial.sh`.
  Running a second `cargo test` while the first is still live is the most
  common self-inflicted "deadlock."
- **The repo root has no `Cargo.toml`.** Each `components/<x>` is its own
  workspace with its own `target/`, so `cd` into the component before a
  scoped `cargo -p <crate>` command.

## Disk

Heavy `--release` runs grow `target/`. Check `du -sh target/` and
`df -h /` before and after; clean per `docs/` disk-hygiene guidance when
`target/` passes ~10 GB. See the disk-aware build hygiene section in
`CLAUDE.md`.
