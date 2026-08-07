# Companion repositories: the sw-MLPL ecosystem

This is a PLANNING document (forward-looking; it names planned
work, unlike the user-facing surfaces). It maps the
repositories that surround `sw-mlpl` -- the ones that exist and
the ones worth building -- and points at a per-repo doc for
each planned effort.

The core `sw-mlpl` repository is the language, interpreter,
web playground, and server. Everything that would bloat that
core, move on a different release cadence, or serve a distinct
audience belongs in a companion repository instead. The
guiding rule mirrors the cellular-monorepo philosophy one
level up: narrow repositories with clear contracts, depending
on `sw-mlpl` through its public surfaces (the `mlpl-repl`
binary, `include` / `run_script`, the connect-mode HTTP API,
the builtin catalog, the typed test-event transport) rather
than reaching into its internals.

## Existing companions

| Repo | What it is | Depends on |
|---|---|---|
| [mlplunit](https://github.com/softwarewrighter/mlplunit) | xUnit-style testing framework for MLPL programs | `mlpl-repl`, structural equality, callables, `@test` reflection, `bracket`, typed test events, `parse_json`, the sandboxed fs API + `run_script` |
| [demo-algorithms](https://github.com/sw-ml-study/demo-algorithms) | Data-structures and algorithms demos in MLPL (general-purpose programming beyond the ML core) | `mlpl-repl`, `include` |
| [demo-combinators](https://github.com/sw-ml-study/demo-combinators) | To Mock a Mockingbird combinatory-logic teaching sequence | first-class references, `call`, partial application |

`mlplunit` is the proving ground that has driven most of the
language's general-purpose surface: each capability it needed
(structural equality, static `include`, first-class callables,
test metadata + reflection, `bracket`, typed events, the fs
API, `run_script`, `parse_json`) landed in `sw-mlpl` because a
real downstream program could not proceed without it. That
feedback loop is the model for the ecosystem: companions state
an executable need, the core grows the smallest surface that
satisfies it, and the capability becomes general rather than
framework-specific.

## Planned companions

Each links to its own planning doc:

- [sw-mlpl-libraries](companion-sw-mlpl-libraries.md) -- a
  curated collection of reusable MLPL modules (the standard
  library that `include` / `run_script` were built to load),
  and the distribution question that comes with it.
- [demo-memory](companion-demo-memory.md) -- memory-
  organization demos bridging classical data structures (hash
  tables, caches, Bloom filters, LRU) to modern ML memory (KV
  caches, sparse attention, retrieval), each self-measuring.
  Brief: `docs/sw-mlpl-demo-memory.txt`.
- [sw-mlpl-lsp](companion-sw-mlpl-lsp.md) -- a Language Server
  Protocol implementation: diagnostics, completion, hover,
  go-to-definition, and test running for any LSP-speaking
  editor.
- [sw-mlpl-mcp](companion-sw-mlpl-mcp.md) -- a Model Context
  Protocol server exposing MLPL evaluation, introspection, and
  test discovery to AI agents as typed tools.

## The strategic thread: a language legible to AI

Several of the planned companions (sw-mlpl-lsp, sw-mlpl-mcp,
and the `swml-explain` / `swml-trace` / `swml-visualize`
libraries) share one bet, drawn from the direction brief
(`docs/sw-mlpl-direction-proactive-response-to-criticisms.txt`):
the project's largest long-term contribution may not be "an AI
built into the language" but **a language designed to be
understood by both humans and AI systems through rich semantic
tooling rather than source text alone.**

Most coding agents today read source text. sw-MLPL is
positioned to expose compiler SEMANTICS instead -- ASTs, typed
intermediate representations, shape information, purity,
dependency graphs, tensor provenance, generated Rust,
optimization opportunities, execution traces -- through clean
LSP and MCP interfaces. Any capable agent (Claude, ChatGPT,
JetBrains AI, Copilot, a local model) then becomes far more
effective without first mastering MLPL syntax. The goal is the
best semantic INTERFACE for AI assistants, not a bespoke
assistant. Both sw-mlpl-lsp and sw-mlpl-mcp are instances of
this thread; their upstream request (structured access to
compiler artifacts) is the same one.

## Other candidates (not yet documented)

- **Editor integrations beyond LSP** -- `mlpl-mode.el` ships in
  this repo under `editors/emacs/`; a VS Code / TextMate
  grammar and a tree-sitter grammar are natural siblings, and
  the LSP above serves all of them.
- **A math-view / literate-notebook surface** -- annotations
  (`@formula`, `@doc`) are already harvestable via
  `annotations(...)`; an org-mode/elisp or web renderer that
  turns them into typeset mathematics is sketched in
  `docs/bqn-sw-mlpl-and-math.txt`.
- **A package registry** -- the distribution half of
  sw-mlpl-libraries, if the library collection outgrows a
  single repository.
- **sw-mlpl-book** -- a long-form tutorial/textbook drawing on
  the web playground's demo and lesson corpus.

## Maturation before wider feedback

The hardening work that should precede a larger community's
criticism -- context-aware error messages that suggest fixes,
AI-agent integration via LSP/MCP/semantic tooling, prepared
answers to the criticisms worth wanting, layered marketing,
and the backend-independent IR -- is planned in
[maturation-plan.md](maturation-plan.md).

## Where these docs live

Companion-repo planning docs are `docs/companion-*.md`. They
are planning artifacts: they may describe intended scope,
sequencing, and status. When a companion repository is
actually created, its own README becomes the source of truth
and the doc here shrinks to a pointer plus the dependency
contract, exactly as the existing-companions table above does.
