# Saga 19: LLM-as-Tool REST Integration (v0.16.0)

## Why this exists

MLPL's REPL has a `:ask` slash-command (Saga 19 preview)
that POSTs to a local Ollama server and prints the
reply. That works for interactive questions, but it is
not callable from inside an `.mlpl` script -- it lives
in the REPL's command table, not the language surface.

Saga 19 ships `llm_call(url, prompt, model)` as a
language-level builtin so MLPL programs can compose
LLM calls alongside `tokenize_bytes`, `adam`,
`experiment`, etc. Cleanly enables the "LLM-as-tool"
idiom: pipe a dataset row through an LLM, store the
reply as a string variable, then feed it to the rest
of the pipeline.

CLI-only surface. The browser cannot reach localhost
Ollama without CORS gymnastics + a server-side proxy;
that is explicitly Saga 21's job.

## Non-goals (deferred)

- **Streaming SSE.** Single POST, whole-reply return.
  Streaming needs a different return shape (string
  chunks or a callback) and is load-bearing for chat
  UIs but not for the compose-in-a-script story.
- **Tool / function-calling JSON schemas.** OpenAI-
  style `tools` + `tool_choice` parameters; requires
  a JSON-schema parser on the MLPL side. Deferred
  until a concrete use case surfaces.
- **Chat threading.** `llm_call` is single-turn.
  Multi-turn conversations need an array-of-messages
  argument; a `llm_chat(history_array, prompt)`
  variant can land later if useful.
- **Request batching.** One call per builtin
  invocation. Batch is easy to add if rate-limit
  pressure appears.
- **In-source auth secrets.** Bearer tokens come from
  env vars (same pattern as Ollama doesn't need it at
  all but OpenAI-compatible endpoints would); never
  accept a literal token as a builtin argument.
- **Teacher-model distillation pipeline.** Uses
  `llm_call` to generate soft labels from a teacher
  model, then distills into a student. Separate saga
  once the builtin is stable.
- **Web / WASM support.** Browser cannot POST to
  localhost Ollama without CORS + a server-side
  proxy. Ship only the CLI path here; the web story
  is Saga 21's CLI server.

## Quality requirements (every step)

Identical to Saga 16 / 16.5; `docs/sw-checklist-
patterns.md` is the decomposition reference.
Design for budgets up front.

## Phase 1 -- language-level builtin (1 step)

### Step 001 -- `llm_call(url, prompt, model) -> string`

1. **Signature**.
   - `url` -- string scalar. Base URL of an
     Ollama-compatible `/api/generate` endpoint
     (or the full endpoint URL; if it lacks the
     `/api/generate` path, append it). Example:
     `"http://localhost:11434"` or
     `"http://localhost:11434/api/generate"`.
   - `prompt` -- string scalar. The user prompt.
   - `model` -- string scalar. Model name
     (`"llama3.2"`, `"qwen2.5-coder"`, etc.).
   - Returns a string scalar: the model's
     completion text. The MLPL string-value machinery
     (Saga 12's `env.get_string` / `set_string` +
     `Value::String`) already exists.
   - Timeout: 120 seconds (same as `:ask`).

2. **HTTP path**.
   - Build request body:
     `{"model": <model>, "prompt": <prompt>,
       "stream": false}`.
   - POST to `<url>` + `/api/generate` (unless the
     URL already ends with it).
   - Parse response JSON; extract `response` field
     as string.
   - Non-200 or missing field -> `EvalError::
     Unsupported("llm_call: <actionable message>")`.

3. **Module**: new
   `crates/mlpl-runtime/src/llm_builtins.rs`. Small
   -- orchestrator + validate + `build_request` +
   `parse_response` + `try_call`. Budget-conscious
   from the start (5 fns, within the 7-fn cap).

4. **Dependency plumbing**.
   - Add `ureq` + `serde_json` to
     `crates/mlpl-runtime/Cargo.toml` (already in
     `mlpl-repl` for `:ask`, but the runtime crate
     does not carry them yet).
   - Add `mockito = "1"` as a dev-dep on
     `crates/mlpl-eval` for tests.

5. **Contract**: `contracts/eval-contract/
   llm-call.md` -- signature, HTTP surface, error
   cases, non-goals (no streaming / tools / chat
   threading / auth / web).

6. **TDD** in
   `crates/mlpl-eval/tests/llm_call_tests.rs`:
   - Happy path. `mockito` stub returns
     `{"response": "hello world"}` for POST
     `/api/generate` with expected body; builtin
     returns the string.
   - URL auto-append: pass a bare
     `http://host/`; builtin POSTs to
     `http://host/api/generate`. Pass a full
     `/api/generate` URL; builtin does not
     double-append.
   - Non-200 response: mock returns 500; builtin
     errors with actionable message.
   - Missing `response` field: mock returns
     `{"foo": "bar"}`; builtin errors.
   - Wrong arity.
   - Non-string argument (e.g., number).

7. Wire `llm_call` into the string-returning
   dispatch path (same path `tokenize_bytes` uses:
   returns `Value::String`, not `Value::Array`).

## Phase 2 -- demo + REPL migration + docs (1 step)

### Step 002 -- demo + `:ask` migration + docs

1. **`demos/llm_tool.mlpl`** (CLI-only). Minimal
   end-to-end: `reply = llm_call("http://
   localhost:11434", "summarize MLPL in one
   sentence", "llama3.2")` followed by inspecting
   the reply. Include a guard comment saying the
   demo needs a running Ollama; show the expected
   output format. Also demonstrate piping the
   reply into `tokenize_bytes` (just to show the
   compose story is real).

2. **Migrate `:ask` to use `llm_call`**. `:ask`'s
   private `call_ollama` helper currently duplicates
   the HTTP path the new builtin will own. Rewrite
   `:ask` to call `llm_call` via `eval_program`
   (simplest) or via a direct function call into
   the runtime (a bit cleaner). DRY the HTTP path.

3. **`docs/using-llm-tool.md`**. Retrospective +
   user guide: the signature, the Ollama setup
   steps, env-var overrides, the `tokenize_bytes`-
   after-`llm_call` idiom, the non-goals list,
   the CLI-only scope (Saga 21 for the web story).

4. Add an "LLM-as-tool" web REPL lesson? **No** --
   the feature is CLI-only, so keep it out of the
   web surface. Mention the CLI-only scope in the
   lesson list's intro text so the web reader knows
   where to look.

5. Update `docs/configurations.md` with an
   `llm_call` row in the CLI-vs-web support matrix.

## Phase 3 -- release (1 step)

### Step 003 -- release v0.16.0

1. Bump `Cargo.toml` workspace.package.version
   `0.15.0 -> 0.16.0`. Minor-level bump because
   new language surface (not a patch).
2. `CHANGELOG.md`: v0.16.0 section above v0.15.0
   (Saga 22's release). Document `llm_call`, the
   demo, the `:ask` migration, the CLI-only
   scope, and the explicit deferred list.
3. `docs/saga.md`: Saga 19 retrospective block
   above Saga 22 (the most recent retrospective)
   to keep reverse-chronological ordering.
4. `docs/status.md`: Saga 19 row moves from Planned
   to Completed; "Next saga to start" pointer
   updates to Saga 21 (CLI server).
5. `cargo build --release` to confirm the bump.
6. `./scripts/build-pages.sh` to pick up any web-
   REPL intro changes, then commit pages/.
7. `./scripts/gen-changes.sh` to refresh CHANGES.md.
8. `/mw-cp` quality gates.
9. Tag `v0.16.0` locally; do NOT push without
   explicit user confirmation (v0.13.0 / v0.14.0
   / v0.14.1 / v0.15.0 cadence).
10. `agentrail complete --done`.

## Dependency graph

```
001 llm_call builtin
        |
002 demo + :ask migration + docs
        |
003 release v0.16.0
```

Each step depends on the previous; sequential.
