# Companion repo: sw-mlpl-mcp

Planning doc (forward-looking). Overview: `companion-repos.md`.

## Purpose

A Model Context Protocol server that exposes MLPL to AI agents
as typed tools: evaluate an expression, run a script, inspect
a value's structure, list the builtins, discover and run
tests. Where sw-mlpl-lsp serves human editors, sw-mlpl-mcp
serves LLM clients (Claude and others) that want to compute
in, and reason about, an array language through a structured
interface rather than by scraping REPL text.

## Why the surfaces line up

MCP tools are request/response over typed data, which is what
the connect-mode server already speaks:

- **Evaluate** -- `mlpl-serve` sessions + `/eval` return typed
  results (kind, shape, values, string lists, viz nodes); an
  MCP `eval` tool is a thin adapter over one session.
- **Run a file** -- `run_script(path, {capture: 1})` returns
  structured status plus captured typed test events; an MCP
  `run_script` tool surfaces that record directly.
- **Introspect** -- `repr` gives bounded deterministic
  rendering, `equal` gives honest comparison, and the builtin
  catalog gives the tool/skill list an agent needs to know
  what it can call.
- **Discover + run tests** -- `tests()` / `test_info(name)` +
  the `--test-events` transport make a `list_tests` / `run_tests`
  tool pair straightforward, returning per-test typed results.
- **Structured errors** -- caught hard errors are already
  `{kind, message}` records, and `parse_json` lets an agent
  consume any JSON the tools emit; both keep the boundary typed
  rather than prose.

## Proposed tool set

```text
eval(program)            -> typed result record
run_script(path, opts)   -> {status, value, events, ...}
describe(name)           -> value / builtin / fn metadata
catalog()                -> builtins with signatures + docs
list_tests(path)         -> stable test names in source order
run_tests(path)          -> per-test typed results
```

## Design notes

- **Isolation.** Each MCP client gets its own server session
  (already the connect-mode model); `run_script`'s fresh
  environment keeps a tool call from leaking definitions into
  the next.
- **Sandbox.** Filesystem tools inherit the `--source-dir`
  sandbox; an MCP deployment sets the root explicitly, so an
  agent cannot escape it -- the same containment the fs API
  enforces.
- **Auth / headless.** `mlpl-serve` already supports
  `--auth`; note that interactively-authenticated transports
  may be absent in headless/cron contexts, so the server
  should degrade to eval-only cleanly.

## The bigger bet: expose compiler semantics as tools

Beyond eval and test-running, the direction brief's central
thesis (`companion-repos.md`) is that MLPL should hand an
agent the compiler's understanding as typed tools -- ASTs,
typed IRs, shape information, purity, dependency graphs,
tensor provenance, generated Rust, optimization opportunities,
execution traces -- so "the best semantic interface for AI
assistants" is the deliverable, not an AI built into the
language. Those artifacts are the same ones sw-mlpl-lsp
surfaces to editors; MCP is their agent-facing projection, and
the two share the upstream request for structured access to
compiler internals.

## Relationship to sw-mlpl

Almost entirely a consumer of the existing server + reflection
surfaces. The likeliest upstream request is the same
machine-readable catalog export that sw-mlpl-lsp wants, reused
here as the `catalog()` tool.
