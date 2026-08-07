# Companion repo: sw-mlpl-libraries

Planning doc (forward-looking). Overview: `companion-repos.md`.

## Purpose

A curated collection of reusable MLPL modules -- the standard
library that the `include` and `run_script` work was built to
make possible. Today a program that wants a z-score, a small
statistics helper, or a combinator aviary either re-defines it
inline or copies a file; a libraries repo turns those into
named, tested, includable modules.

## Two layers, two kinds of learning

The design brief
(`docs/sw-mlpl-direction-proactive-response-to-criticisms.txt`)
draws a distinction worth building the collection around:

```text
sw-MLPL
    language
        |
    standard library          (general-purpose modules)
        |
    agent-experimentation library
```

It maps onto TWO KINDS OF LEARNING the platform supports:

- **Machine learning** -- learning by gradient descent (train
  a network). The traditional reading; the ML core already
  serves it.
- **Agent learning** -- learning by EXPERIENCE: reflection,
  retry, tool use, memory, retrieval, planning, self-critique,
  skill acquisition, workflow optimization. Increasingly
  central to modern systems, and the brief argues it deserves
  first-class treatment -- but as LIBRARIES, not language
  keywords, so the core stays small and the ideas stay
  experimental and swappable.

The standard library is the stable, general-purpose layer; the
agent-experimentation library is where memory / ICL / ICRL /
reflection loops live as composable modules. Keeping the
second a library (not built-in syntax) is deliberate: these
are experiments ABOUT improving agents, and experiments should
not calcify into keywords.

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

## A proposed library set

The direction brief sketches a concrete set (its `swml-*`
names; whether they become one repo of modules or several
repos is the distribution question above):

| Priority | Library | Role |
|---|---|---|
| High | swml-core | the language itself (this repo) |
| High | swml-lsp | semantic language server (`companion-sw-mlpl-lsp.md`) |
| High | swml-mcp | AI tool interface (`companion-sw-mlpl-mcp.md`) |
| High | swml-visualize | graphs / diagrams as a library surface |
| High | swml-explain | compiler explanations (errors, suggestions) |
| High | swml-trace | runtime inspection |
| Medium | swml-ai | adapter layer to external AI coding agents |
| Medium | swml-rust | generated-Rust inspection |
| Later | swml-agents | reflection / memory / ICL / ICRL -- the agent-experimentation layer above |

The "high" tier is mostly SEMANTIC TOOLING rather than
algorithm code -- which is the strategic bet spelled out in
`companion-repos.md` and the LSP/MCP docs: expose the
compiler's understanding (ASTs, shapes, purity, traces,
generated Rust) so any coding agent works better, rather than
building an AI into the language.

## Relationship to sw-mlpl

Pure consumer of public surfaces -- no core changes required
for the MVP. The namespacing question (1) is the one likely
source of an upstream request, and it should follow the
mlplunit model: state the executable need, let the core grow
the minimal surface. The semantic-tooling libraries
(swml-lsp / swml-mcp / swml-explain / swml-trace) share one
upstream need: structured access to compiler artifacts, tracked
in their own docs.
