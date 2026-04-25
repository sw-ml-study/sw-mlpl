Phase 2 step 002: demo + `:ask` migration + docs.

With `llm_call` shipped (step 001), integrate it into
the demo surface and DRY the REPL's HTTP path.

1. **`demos/llm_tool.mlpl`** (CLI-only). Minimal
   end-to-end script that exercises `llm_call`:
   - Call Ollama's default model with a short
     prompt.
   - Print the reply.
   - Feed the reply into `tokenize_bytes` to show
     composition works.
   - Guard comment at the top: "Requires a
     running Ollama server at localhost:11434 with
     `llama3.2` pulled. Skip this demo on CI /
     no-network envs."
   - Show expected output format at the bottom of
     the file.
   - Run-guide line: `mlpl-repl -f
     demos/llm_tool.mlpl`.

2. **Migrate `:ask`** (`apps/mlpl-repl/src/ask.rs`).
   `:ask`'s private `call_ollama` currently has its
   own ureq POST. Rewrite to call the language-
   level `llm_call` via `eval_program`: construct
   the MLPL source string with the prompt safely
   escaped, call `eval_program`, extract the
   resulting string. If the escape story is gnarly,
   alternative: call into `llm_builtins` directly
   as a `pub(crate)` helper. Either way, the HTTP
   path lives in ONE place.
   - The existing `build_system_prompt` +
     `build_user_context` + `var_summary` helpers
     stay where they are (`:ask`-specific
     framing); only the POST gets replaced.
   - `:ask` still uses `/api/chat` (multi-turn
     system + user role), while `llm_call` uses
     `/api/generate` (single-turn). Keep the
     distinction: `:ask` concatenates
     system+user context into one string and
     passes it to `llm_call`; the model loses the
     role distinction but the context is
     preserved. Note this in ask.rs as a comment
     with a followup TODO referencing a future
     `llm_chat` variant (step 001's deferred
     list).

3. **`docs/using-llm-tool.md`**. Retrospective +
   user guide. Sections:
   - Status block (shipped in Saga 19 / v0.16.0).
   - What this is about (CLI-only language-level
     LLM calls).
   - The builtin (signature, return type,
     Ollama setup, env-var overrides; link to
     contract).
   - Composition story (`tokenize_bytes` on the
     reply, `experiment {}` wrapping an LLM
     sweep, etc.).
   - The `:ask` relationship (REPL command uses
     `llm_call` under the hood with a
     system+user framing).
   - Why no web / WASM support (browser + CORS +
     Saga 21's CLI server).
   - Non-goals / deferred list (streaming, tools,
     chat threading, batching, distillation).

4. **`docs/configurations.md`**. Add an
   `llm_call(...)` row in the CLI-vs-web matrix
   with "CLI-only, needs local Ollama" in the
   notes column.

5. Do NOT add a web-REPL lesson. The feature is
   CLI-only and browser users would just get a
   broken demo. Keep the web REPL focused on
   pure in-process features.

6. Quality gates + commit. Commit message
   references Saga 19 step 002.
