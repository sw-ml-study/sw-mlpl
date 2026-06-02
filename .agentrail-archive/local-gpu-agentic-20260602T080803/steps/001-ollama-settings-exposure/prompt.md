Phase 0 of the local-gpu-agentic saga (see docs/saga-local-gpu-agentic.md).

Goal: make the Ollama host + model first-class and discoverable on
mlpl-serve, replacing the ?ollama= / ?model= query-string stopgap.

Scope:
- Server-owned default Ollama host + default model on mlpl-serve,
  configurable via a config file and/or CLI flags and/or the
  OLLAMA_HOST env var. The browser should not have to carry the host.
- A model listing: GET <host>/api/tags surfaced as a `:models ollama`
  REPL command (distinct from the existing MLPL `:models`) and/or a
  small UI picker, so the user can see which Ollama models are
  available on the configured host.
- The web `:ask` should default to the server-configured host/model
  when no override is given.

Explicitly DEFERRED (user request): the per-`:ask --model <name>`
override syntax. Do NOT build it this step.

Policy (from the saga plan): outbound Ollama hosts are allow-listed in
server config (mirrors the connect-mode CORS allow-list); no arbitrary
network from the web/WASM build.

TDD: write failing tests first (config parse/precedence; /api/tags
listing handler; :models ollama formatting). Keep sw-checklist
non-regressing (retire warnings/FAILs where the new code lands).
Rebuild + restart mlpl-serve and rebuild pages/ since the web :ask
path and server config change. Commit .agentrail/ with the source.
