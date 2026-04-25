# `mlpl-repl --connect <url>` Contract (Saga 21 step 002)

## Purpose

`--connect <url>` turns `mlpl-repl` into a thin
client of a remote `mlpl-serve`: every line the user
types either dispatches a slash command (handled
locally OR via a `GET /inspect` round-trip) or POSTs
to `/eval`. The local `mlpl_eval::Environment` is
unused -- the server holds all state across the
session lifetime.

This is the first real client of the
`crates/mlpl-serve` REST surface. Future clients
(ratatui TUI, Emacs, desktop GUI, web UI re-routing)
will follow the same contract.

## Flag

```
mlpl-repl --connect <base-url>
```

`<base-url>` is the bare server origin -- the
`/v1/...` path prefix is appended by the client.
Trailing slashes are stripped before path
construction, so `http://host:6464` and
`http://host:6464/` are equivalent.

## Local-mode-only flag interactions

`--connect` is incompatible with these flags
(they all assume a local `Environment`):

- `-f` / `--file` (script-mode local execution)
- `--data-dir` (local sandbox for `load(...)`)
- `--exp-dir` (local on-disk experiment records)

If any are combined with `--connect`, the REPL
prints an actionable error and exits with code 2:

```
error: --connect cannot be combined with -f
  --connect delegates evaluation to a remote
  server; -f, --data-dir, and --exp-dir are
  local-mode only.
```

Compatible flags (carried through unchanged):

- `--version` / `-V` (prints client version, no
  server contact).

## Lifecycle

1. **Startup** -- POST `<url>/v1/sessions` (no
   auth needed for this call). Store
   `(session_id, token)` for the duration of the
   session. If the call fails, the REPL prints an
   actionable hint pointing at `mlpl-serve` and
   exits.
2. **Per line** -- if the line starts with `:`,
   dispatch as a slash command (see below).
   Otherwise POST `<url>/v1/sessions/{id}/eval`
   with `{program: <line>}` and the bearer token;
   print the returned `value` or the unwrapped
   `error` string.
3. **Shutdown** -- `exit` or `Ctrl-D` exits the
   client. The remote session stays in the
   server's memory; restarting the client with
   the same token would reattach (not exposed in
   the client today, but possible from the API).

## Slash commands

These are handled in connect mode:

| Command | Source | Notes |
|---|---|---|
| `:vars` | `GET /inspect` | Renders the snapshot's `vars` list (sorted by name, with shape + `[param]` tag). `more: <n>` line if the server truncated. |
| `:models` | `GET /inspect` | Sorted name list. |
| `:tokenizers` | `GET /inspect` | Sorted name list. |
| `:experiments` | `GET /inspect` | Deduplicated sorted name list. |
| `:wsid` | `GET /inspect` | Counts (variables include the truncated tail; models / tokenizers / experiments are exact). |
| `:ask <q>` | local `mlpl_runtime::call_ollama` | See "`:ask` carve-out" below. |
| `:help` | client-static | Prints the command list. |

Other slash commands available in local mode
(`:trace on`, `:clear`, `:describe`, `:builtins`,
`:fns`, etc.) print
`<command>: not supported in --connect mode (try
:vars, :models, :experiments, :tokenizers, :wsid,
:ask, :help)` rather than being POSTed as MLPL
source. They land in follow-up sagas as the inspect
endpoint grows.

## `:ask` carve-out

`:ask` does NOT route through the server. It calls
`mlpl_runtime::call_ollama` directly with the local
`OLLAMA_HOST` and `OLLAMA_MODEL` env vars (defaults:
`http://localhost:11434`, `llama3.2`). The
**server's workspace is NOT threaded into the
prompt** in connect mode -- that would require
composing the inspect snapshot into `ask.rs`'s
helpers, which is a follow-up.

This carve-out exists for two reasons:

1. **The server-side LLM proxy is post-MVP.** The
   eventual path is "browser / connect-mode client
   asks server, server hits an allow-listed Ollama
   on the user's behalf." Until that ships, the
   client talks to its local Ollama directly.
2. **The framing is workspace-shape-aware.** Today
   `:ask` reads the local `Environment` to build
   the workspace summary. Without the local env in
   connect mode, the system prompt has nothing to
   ground itself on -- so the connect-mode `:ask`
   simply forwards the question, no framing.

## Networking

- **HTTP client:** `reqwest::blocking::Client`
  with a 120-second timeout (matches `:ask` and
  the language-level `llm_call`). Default-features
  off + `rustls-tls` so the REPL crate doesn't
  pull `tokio` for the client side.
- **Bearer auth:** every `/eval` and `/inspect`
  call sends `Authorization: Bearer <token>` from
  the session-create response.
- **Errors:** the client distinguishes
  `ClientError::Network(...)` (transport failures
  -- DNS, refused, timeout) from
  `ClientError::Server { status, message }` (4xx
  / 5xx with the unwrapped `error` field from the
  JSON body when present, else the raw body).

## Module layout

- `apps/mlpl-repl/src/connect.rs` (7 fns at the
  per-module cap):
  `create_session`, `eval_remote`,
  `inspect_remote`, `read_loop`, `dispatch_slash`,
  `format_inspect`, plus the `Display` impl on
  `ClientError`.
- `apps/mlpl-repl/src/main.rs` -- `--connect`
  parsing happens before the local-mode setup so
  the local `Environment` is never constructed
  in connect mode.

## Non-goals (deferred)

- **`-f` over `--connect`.** Running an `.mlpl`
  script through a remote server. Useful but
  needs a different command-input shape (whole-
  file vs line-at-a-time); follow-up.
- **`:trace`.** The MVP server doesn't expose
  tracing; the connect-mode `:trace` slash
  commands print "not supported."
- **`:describe <name>`.** Needs a per-name lookup
  endpoint (`GET /v1/sessions/{id}/describe/<name>`)
  that step 002 doesn't ship.
- **Server-side `:ask` framing.** See the carve-
  out section. Once the LLM proxy ships, the
  client `:ask` will switch to the proxied path.
- **Reattach / token reuse across REPL restarts.**
  Sessions persist on the server but the client
  generates a new one on every connect. A
  `--session <id>` flag could re-attach if the
  user has the token; not shipped today.
- **Cancellation / interrupt of in-flight evals.**
  Server doesn't expose it; client can't ask for
  it.
