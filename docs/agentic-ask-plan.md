# Agentic `:ask` -- Saga Plan (tool-using, RLM-like)

## Vision

Today `:ask` sends one context-rich prompt to Ollama. The goal is
to make it **agent-like**: a tool-capable model is given a *meta
context* describing what context is *available on request*, and it
**requests** only what it needs (recent history, a variable's
shape/values, the selected sculpture, a builtin's help, a demo's
source) via tool calls, then answers from what it retrieved. This
is the start of Recursive-Language-Model-style tooling -- the model
drives a retrieval/answer loop instead of being flooded up front.

## Already shipped (the foundation this builds on)

- **Connect-mode evaluator swap (SSE).** `?connect=<url>` routes
  REPL eval to `mlpl-serve`, executed server-side (no browser CORS,
  no WASM `std::time` panic). `connect.rs` + `eval_wasm::connect_eval`.
- **Contextual `:ask`.** Prompt = a "you are inside the sw-MLPL
  REPL" system preamble + recent REPL activity + the selected 3D
  sculpture summary (`window.__stage3d_context()`). Plus `:history`,
  the in-dialog **Ask** button (`window.__mlpl_ask`), and
  configurable `?ollama=` / `?model=`.
- **Ollama tool-calling verified.** Capable models (`llama3.1`,
  `qwen2.5-coder:14b`) return proper `tool_calls`; the tiny
  `qwen2.5:0.5b` emits the call as plain text -- so the agent loop
  needs a strong model (e.g. `qwen2.5-coder:14b` on the LAN box).

## Phase 0 -- Ollama settings exposure (prerequisite)

Make host + model first-class and discoverable rather than
hard-coded / query-string-only:

- **Host:** `localhost:11434` or a LAN host (e.g.
  `http://large12:11434`). The call is server-side, so any host
  reachable from `mlpl-serve` works.
- **Server-owned config:** a `config.toml` (and/or CLI flags /
  `OLLAMA_HOST` env) on `mlpl-serve` holding the default host +
  default model, so the browser carries nothing.
- **List models:** `GET <host>/api/tags` -> a model picker in the
  UI; a `:models ollama` listing (distinct from MLPL `:models`).
- **Default + override:** choose a default model; override per call
  (`:ask --model qwen2.5-coder:14b <question>` and/or a UI
  dropdown). Replaces the current `?ollama=`/`?model=` stopgap.

## Phase 1 -- proper system/user messages

Switch from `/api/generate` with the system text embedded in the
prompt to `/api/chat` with a real `system` message + `user`
message (Ollama supports both). Weak models follow a real system
role far better. Extend `llm_call` (optional `system` arg) or add a
server-side `ask()` builtin.

## Phase 2 -- the tool-using agent loop

Run the loop **server-side** (it owns the live session):

1. `POST <host>/api/chat` with `messages` + `tools` (function defs)
   + the meta-context system prompt ("you have these tools; request
   only what you need; do not guess").
2. If the reply has `tool_calls`, fulfill each from the session and
   append the results as `tool` messages; loop (bounded depth /
   token budget).
3. When the model returns content with no tool calls, that's the
   answer.

**Context-provider tools** (map to existing introspection):

| tool | source |
|------|--------|
| `get_recent_history()` | REPL history (client snapshot) |
| `get_workspace_vars()` | session env (`:vars`) |
| `describe_variable(name)` | `:describe` / inspect |
| `get_selected_sculpture()` | client `__stage3d_context()` snapshot |
| `get_builtin_help(name)` | `:help` table |
| `get_demo_source(name)` | the demo registry |

Client-only context (the 3D selection) is passed as a snapshot in
the initial request so the server can answer those tool calls
without a round-trip to the browser. Surface the tool-call trace in
the UI for transparency ("the model looked at: history, var X").

## Phase 3 -- the RLM demo

A demo that **runs ML steps, then asks for help**:

1. Train a tiny model (or build an attention pattern) -- a few REPL
   lines.
2. Issue an agentic `:ask` ("why did the loss plateau?" / "what does
   this attention pattern show, given my setup?").
3. The model calls `get_recent_history` + `get_workspace_vars` +
   `describe_variable`, then answers from what it retrieved -- the
   visible RLM-like loop.

## Open questions / decisions

- **Model requirement:** tool-capable; default to a strong remote
  one (`qwen2.5-coder:14b` on `large12`). Document the floor.
- **Where the loop lives:** server-side; client passes the 3D
  selection snapshot. (A future variant could let the server call
  back to the client for live UI state.)
- **Flooding control:** per-tool result size caps; the point of the
  meta-context is selective retrieval.
- **Security:** the server makes outbound calls -- allow-list the
  Ollama host(s) in config (mirrors the connect-mode CORS
  allow-list).
- **Live-demo gating:** these features only work in connect mode;
  the GitHub Pages demo should mark them visible-but-not-runnable
  (a `requires_connect` flag on the demo registry -- separate small
  task).

## References

- `docs/using-ollama.md`, `docs/using-cli-server.md` -- LLM + server.
- `components/web-handlers/.../connect.rs` -- current `:ask`.
- `components/runtime-core/.../llm_builtins.rs` -- `call_ollama`.
- Recursive Language Models (external research) -- the inspiration.
