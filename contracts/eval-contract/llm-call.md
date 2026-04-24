# `llm_call` Contract (Saga 19 step 001)

## Purpose

`llm_call(url, prompt, model) -> string` is a
language-level builtin that issues a single
synchronous POST to an Ollama-compatible
`/api/generate` endpoint and returns the model's
completion text as a string scalar. It enables the
"LLM as a tool" idiom: pipe a value through a hosted
LLM, store the reply as an MLPL string, then feed
it to the rest of the pipeline alongside
`tokenize_bytes`, `experiment`, etc.

```mlpl
reply = llm_call("http://localhost:11434",
                 "summarize MLPL in one sentence",
                 "llama3.2")
tokens = tokenize_bytes(reply)
```

CLI-only. The browser cannot reach localhost
Ollama without CORS gymnastics + a server-side
proxy; that is Saga 21's job.

## Signature

```
llm_call(url: string, prompt: string, model: string)
  -> string
```

- **`url`** -- string scalar. Ollama base URL
  (`"http://localhost:11434"`) or the full endpoint
  (`"http://localhost:11434/api/generate"`). Trailing
  slashes are stripped before the path is appended.
  If the resolved URL does not end with
  `/api/generate`, the suffix is appended; otherwise
  used verbatim (no double-append).
- **`prompt`** -- string scalar. The user prompt sent
  in the request body.
- **`model`** -- string scalar. Ollama model name
  (`"llama3.2"`, `"qwen2.5-coder"`, etc.).
- **Returns** -- string scalar (`Value::Str`)
  containing the model's completion text from the
  response's `response` field.
- **Timeout** -- 120 seconds. Matches the `:ask`
  REPL command. A streaming variant could ship later
  with a different return shape; this builtin is
  the single-POST whole-reply form.

## HTTP surface

Request body (JSON):

```json
{"model": "<model>", "prompt": "<prompt>", "stream": false}
```

POSTed with `Content-Type: application/json` to the
resolved URL. Response is parsed as JSON; the string
at the top-level `response` field becomes the return
value. `stream: false` is hardcoded -- streaming SSE
is a deferred non-goal.

## Errors

`llm_call` returns errors via the runtime/eval error
chain (surfacing through `EvalError::Unsupported`
when called from MLPL source), with a message that
identifies the function name `llm_call`:

- **Wrong arity.** `llm_call` called with anything
  other than 3 arguments yields
  `EvalError::BadArity { func: "llm_call", ... }`.
- **Non-string argument.** Any of the three
  positional args evaluating to a non-string
  (`Value::Array`, `Value::Model`, `Value::Tokenizer`)
  yields `EvalError::ExpectedString`.
- **Connection failure.** The `ureq` agent can't
  reach the host (refused, DNS, timeout). Error
  message includes the URL and the underlying
  `ureq::Error` text.
- **Non-2xx status.** Server returned a status code
  outside 200-299. Error message contains
  `"llm_call"`, the status code, and a 200-character
  preview of the response body.
- **Invalid JSON.** Response body did not parse as
  JSON. Error message includes the URL.
- **Missing `response` field.** Response JSON did
  not contain a string at top-level `response`.
  Error message includes the parsed JSON for
  debugging.

## Non-goals (deferred)

- **Streaming SSE.** Single POST, whole-reply
  return. Streaming needs a different return shape
  (chunk array or callback) and is load-bearing for
  chat UIs but not for the compose-in-a-script
  story. Future variant: `llm_stream(...)` or a
  callback parameter.
- **OpenAI-style tool calling / function schemas.**
  `tools` + `tool_choice` request fields plus the
  associated JSON-schema parser on the MLPL side.
  Deferred until a concrete use case surfaces.
- **Multi-turn chat threading.** `llm_call` is
  single-turn (one prompt, one reply). Multi-turn
  conversations need an array-of-messages argument;
  a `llm_chat(history_array, prompt)` variant can
  land later. The REPL's `:ask` command stays on
  Ollama's `/api/chat` path because it composes
  workspace context into a system + user message
  pair; the language-level `llm_call` is the
  simpler `/api/generate` form.
- **Request batching.** One call per builtin
  invocation. Batch is straightforward to add if
  rate-limit pressure appears.
- **In-source auth secrets.** The builtin accepts
  no bearer token / API key argument. For
  OpenAI-compatible endpoints, secrets must come
  from process env vars, never from MLPL source.
  Ollama itself does not require auth.
- **Web / WASM support.** Browser cannot POST to
  localhost Ollama without CORS allow-listing +
  a server-side proxy. The CLI-only scope here is
  intentional; the browser story is Saga 21's
  CLI server with a server-side reverse proxy.
- **Teacher-model distillation pipeline.** Uses
  `llm_call` to generate soft labels from a hosted
  teacher, then distills into a student via
  existing `cross_entropy` + `adam`. Separate saga
  once the builtin is stable.

## Module layout

- `crates/mlpl-runtime/src/llm_builtins.rs` -- the
  pure HTTP path (no `Expr` / `Environment`
  knowledge): `call_ollama(url, prompt, model)`
  orchestrator, `resolve_url`, `parse_response`. 3
  functions total, well within the 7-fn module
  budget (`docs/sw-checklist-patterns.md`).
- `crates/mlpl-eval/src/llm_dispatch.rs` -- the
  dispatch shim that evaluates string arguments via
  `eval_expr`, calls the runtime helper, and wraps
  the reply in `Value::Str`. One function (`dispatch`)
  to stay under the eval crate's per-module fn
  budget. Wired into `eval::eval_expr` next to the
  other string-returning dispatches (`tokenizer::
  dispatch`, `loader::eval_load`).

The split mirrors Saga 22's revised layout:
pure-math / pure-IO helpers in `mlpl-runtime`,
`Expr`-aware dispatchers in `mlpl-eval`. The
original step plan named the runtime module's
function set as
`try_call / builtin_llm_call / validate_args /
resolve_url / parse_response`; the actual shipped
split moves `try_call` and `validate_args` into
the eval-side dispatcher because they need
`Expr` and `Environment` access that the runtime
crate intentionally avoids.
