# Shared `target/` infrastructure (saga 51)

Set up a single shared `target/` directory at the repo root so all
workspaces (main + each component) write to the same build cache.
This enables the upcoming component-migration sagas (52+) without
exploding disk usage.

## Steps

1. Create `.cargo/config.toml` at repo root with
   `[build] target-dir = "target"`. Verify `cargo check` from
   main workspace + from `components/mlpl-session/` and
   `services/mlpl-mlx-serve/` all resolve to the same target dir.
   `du -sh target/` before and after should not grow significantly.
2. language-status update + saga close.

## Why one saga for this?

The shared-target config is the prerequisite for every component
migration. It lives by itself in saga 51 so the migration sagas
(saga 52 onward, one per component) start from a clean baseline.
