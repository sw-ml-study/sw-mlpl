Phase 1 step 001: `llm_call(url, prompt, model) -> string`
language-level builtin.

The REPL's `:ask` slash-command (Saga 19 preview) ships
the HTTP path today, but only callable interactively.
Saga 19 ships the language-level form so `.mlpl`
scripts can compose LLM calls alongside
`tokenize_bytes`, `adam`, `experiment`, etc. This step
lands ONLY the builtin; step 002 wires it into the
demo, migrates `:ask`, and writes docs.

1. **Signature**.
   - `url` -- string scalar. Ollama-compatible base
     URL (`"http://localhost:11434"`) or the full
     endpoint (`"http://.../api/generate"`). If the
     URL doesn't already end with `/api/generate`,
     append it (strip trailing slashes first so
     `"http://h/"` still works).
   - `prompt` -- string scalar. User prompt.
   - `model` -- string scalar. Model name.
   - Returns a string scalar (`Value::String`) with
     the model's completion text.
   - Timeout: 120 seconds (match `:ask`).

2. **HTTP path**.
   - Build body: `{"model": ..., "prompt": ...,
     "stream": false}` via `serde_json::json!`.
   - POST to the resolved URL with
     `Content-Type: application/json`.
   - Parse response JSON; pull `response` field as
     string. Non-string or missing -> error.
   - Non-2xx status -> error with the status code
     and a truncated body preview.
   - `ureq::Error` -> error with URL + inner
     message so the user sees the actual connection
     failure.

3. **Module**: new
   `crates/mlpl-runtime/src/llm_builtins.rs`.
   Budget-conscious from the start: 5 functions
   (`try_call` dispatch, `builtin_llm_call`
   orchestrator, `validate_args`, `resolve_url`,
   `parse_response`). Stay under the 7-fn cap per
   `docs/sw-checklist-patterns.md`.

4. **Dependency plumbing**.
   - Add `ureq = { version = "2", default-features =
     false, features = ["json"] }` and `serde_json
     = "1"` to `crates/mlpl-runtime/Cargo.toml`.
     `mlpl-repl` already uses these so no new
     workspace entries are needed; add them as
     direct deps on the runtime crate.
   - Add `mockito = "1"` as a dev-dep on
     `crates/mlpl-eval/Cargo.toml`.

5. **Wire into dispatch**. `llm_call` returns a
   string, so the dispatch site lives where
   `tokenize_bytes` / `decode` etc. live --
   `crates/mlpl-eval/src/eval.rs` (or
   `tokenizer.rs` dispatcher) -- NOT the array
   `call_builtin` chain. Follow the same pattern:
   a new `FnCall` branch that evaluates args to
   strings, calls the runtime helper, and returns
   `Value::String`.

6. **Contract**: new
   `contracts/eval-contract/llm-call.md` --
   signature, HTTP surface (Ollama `/api/generate`
   shape), return type, error cases, non-goals
   (no streaming, no tools, no chat, no auth in
   args, no WASM).

7. **TDD** (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-eval/tests/llm_call_tests.rs`:
   - **Happy path.** `mockito` server returns
     `{"response": "hello"}` for POST
     `/api/generate`; builtin returns `"hello"`.
     Assert request body contains the expected
     model + prompt.
   - **URL auto-append.** Same body but passing the
     mock's base URL (without `/api/generate`);
     builtin still hits the right path.
   - **Full-URL passthrough.** Pass
     `<base>/api/generate` explicitly; no double-
     append.
   - **Non-200.** Mock returns 500 with body
     `"boom"`; builtin errors with message
     containing `"llm_call"`, `"500"`, and
     `"boom"`.
   - **Missing `response` field.** Mock returns
     `{"foo": "bar"}`; builtin errors.
   - **Wrong arity.** 2 args instead of 3.
   - **Non-string argument.** Passing a number
     array where a string is expected errors.

8. Quality gates + commit. Commit message
   references Saga 19 step 001.
