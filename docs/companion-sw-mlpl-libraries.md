# Companion repo: sw-mlpl-libraries

Planning doc (forward-looking). Overview: `companion-repos.md`.

## Purpose

A curated collection of reusable MLPL modules -- the standard
library that the `include` and `run_script` work was built to
make possible. Today a program that wants a z-score, a small
statistics helper, or a combinator aviary either re-defines it
inline or copies a file; a libraries repo turns those into
named, tested, includable modules.

## Why this is possible now

The core capabilities a library ecosystem needs already ship:

- **`include "path.mlpl"`** -- static, sandboxed, source-order
  composition: the mechanism for pulling a module into a
  program.
- **First-class callables + partial application** -- libraries
  can export functions as values, registries as records, and
  combinators that compose them.
- **`@test` metadata + reflection + `bracket`** -- every module
  can ship its own tests, discoverable with `tests()` and
  runnable by mlplunit, with fixture lifecycle guaranteed.
- **The sandboxed fs API + `run_script`** -- a library's test
  runner can walk its own tree and execute modules in fresh
  environments.

## Proposed shape

```text
sw-mlpl-libraries/
    std/            # stats, math helpers, string utilities
    combinators/    # the aviary as an importable module
    result/         # Result-pipeline helpers over ?/map_ok/...
    data/           # dataset prep, splits, normalization
    each/           # tests alongside sources, @test-annotated
```

A module is an ordinary `.mlpl` file whose top level is
`def u:` definitions (plus `@test` functions); a consumer
does `include "std/stats.mlpl"` and calls `u:zscore(v)`.

## Open questions

1. **Namespacing.** MLPL user functions share one `u:` space.
   Two libraries defining `u:mean` collide. Options: a
   convention (`u:stats_mean`), a per-include prefix
   declaration, or leaving late-binding-by-name as the
   documented hazard. This is the first design decision and
   likely feeds a small language change back into the core.
2. **Distribution.** A single repo of `include`-able files is
   the MVP; a package registry + a `--source-dir` that
   resolves package names is the growth path (see the
   registry candidate in `companion-repos.md`).
3. **Versioning.** Modules that ship tests can pin against a
   `sw-mlpl` version via a capability probe rather than a
   version string.

## Relationship to sw-mlpl

Pure consumer of public surfaces -- no core changes required
for the MVP. The namespacing question (1) is the one likely
source of an upstream request, and it should follow the
mlplunit model: state the executable need, let the core grow
the minimal surface.
