# Using MLPL with a Hosted LLM (`llm_call`)

> **Status:** reference. Shipped in Saga 19 (v0.16.0).

## What this is about

The `:ask <question>` REPL command (shipped as a
preview in Saga 14) was the right idea, wrong shape:
it lived in the CLI's command table, so a `.mlpl`
script could not call out to a hosted LLM as part of
its own pipeline. Saga 19 fixes that by promoting
the HTTP path to a language-level builtin --
`llm_call(url, prompt, model)` -- so MLPL programs
can compose hosted-LLM calls alongside
`tokenize_bytes`, `experiment {}`, `adam`, etc.

The headline use case is "LLM as a tool": pipe a
value through a hosted model, store the reply as an
MLPL string, and feed that reply into the rest of
the pipeline. Concrete shapes the builtin enables:

- **Summarization sweep.** Run `experiment "summary-
  v1" { reply = llm_call(...); ... }` over multiple
  prompts, log the replies, compare runs.
- **Synthetic data generation.** Use the LLM to
  produce a small Q/A corpus, then tokenize +
  fine-tune a tiny student model with the existing
  Saga 13 pipeline.
- **Tool-augmented inference.** Use the LLM as the
  reasoning surface and dispatch into MLPL builtins
  for the numerical work (this is the seed for
  Saga 18's distillation + multi-model
  orchestration sagas).

CLI-only. The browser cannot POST to a localhost
Ollama without CORS allow-listing + a server-side
proxy; that path lands in Saga 21 (`mlpl-serve`).
See `docs/configurations.md` for the full
CLI-vs-web matrix.

## The builtin

```
llm_call(url, prompt, model) -> string
```

- **`url`** -- string scalar. Ollama base URL
  (`"http://localhost:11434"`) or the full endpoint
  (`"http://.../api/generate"`). Trailing slashes
  are stripped; if the URL does not already end
  with `/api/generate`, it is appended. No
  double-append.
- **`prompt`** -- string scalar. The user prompt.
- **`model`** -- string scalar. Ollama model name
  (`"llama3.2"`, `"qwen2.5-coder"`, etc.).
- **Returns** -- string scalar (`Value::Str`)
  containing the model's completion text.
- **Timeout** -- 120 seconds. Same as the `:ask`
  preview.

The full signature, error catalog, and HTTP body
shape are pinned in
`contracts/eval-contract/llm-call.md`.

## Setup

One-time: start Ollama and pull a model that fits
on your machine. `llama3.2` (2B params) is the MLPL
default because it loads fast and runs on a laptop
without a discrete GPU.

```bash
ollama serve &
ollama pull llama3.2
```

Verify the language surface from the CLI REPL:

```bash
mlpl-repl
mlpl> reply = llm_call("http://localhost:11434", "Say hi.", "llama3.2")
mlpl> reply
```

The end-to-end demo at `demos/llm_tool.mlpl` runs
the same call and pipes the reply through
`tokenize_bytes` to show that the result is just an
ordinary MLPL string.

## Configuration

The builtin reads its arguments verbatim -- there are
no implicit env-var lookups inside `llm_call`. For
the REPL's `:ask` command (which uses `llm_call`
under the hood), two env vars override the defaults:

- **`OLLAMA_HOST`** -- default
  `http://localhost:11434`. Change to point at a
  remote Ollama or a containerized one.
- **`OLLAMA_MODEL`** -- default `llama3.2`. Any
  model you have pulled with `ollama pull <name>`.

For the language-level form, just pass the values
explicitly. A common pattern:

```mlpl
host = "http://localhost:11434"
mdl = "llama3.2"

prompt_a = "Summarize MLPL in one sentence."
reply_a = llm_call(host, prompt_a, mdl)

prompt_b = "Now summarize NumPy in one sentence."
reply_b = llm_call(host, prompt_b, mdl)
```

## Composition stories

`llm_call` returns a plain MLPL string, so anything
that already accepts a string works.

### Reply -> tokens -> downstream pipeline

```mlpl
reply = llm_call(host, "Generate a short Q/A pair.", "llama3.2")
toks  = tokenize_bytes(reply)
toks
# [N] f64 array of byte values, ready for embed lookup or BPE.
```

### LLM call inside `experiment {}`

```mlpl
experiment "summary-sweep-v1" {
    rep_a = llm_call(host, "summarize MLPL", "llama3.2")
    rep_b = llm_call(host, "summarize NumPy", "llama3.2")
    # Both replies are bound in the experiment record;
    # `:experiments` will show them under this name and
    # `compare(...)` can diff them across runs.
}
```

### As a teacher for distillation (deferred saga)

Once the dtype machinery for fp16/bf16 ships, the
intended pattern is:

```mlpl
# Pseudocode -- distillation pipeline is a separate
# future saga. The Saga 19 builtin is the prerequisite
# the saga waits on.
soft_labels = []
for i in iota(N) {
    reply = llm_call(host, prompt[i], "llama3.2")
    soft_labels = append(soft_labels, encode(reply))
}
loss = cross_entropy(student.apply(X), soft_labels)
adam(loss, student, ...)
```

## Relationship to `:ask`

The CLI REPL's `:ask <question>` command predates the
language-level builtin (it shipped as the Saga 19
preview in v0.11.x). After Saga 19 step 002 it
delegates to `mlpl_runtime::call_ollama` -- the
exact path `llm_call` uses -- so the HTTP machinery
lives in one place.

Two things `:ask` still does on top of the raw
builtin:

1. **Workspace context framing.** It builds a
   compact `:vars`-style summary (variable names +
   shapes + `[param]` tags, capped at 40 entries),
   prepends it to the user's question, and adds a
   short "you are a concise assistant inside MLPL"
   system header.
2. **Endpoint downgrade from `/api/chat` to
   `/api/generate`.** The pre-v0.16 `:ask` used
   Ollama's chat endpoint with explicit
   `{role: system}` and `{role: user}` messages.
   The new `:ask` concatenates the same content
   into a single prompt string and calls
   `/api/generate`. The model loses the role
   distinction but keeps the context. A future
   `llm_chat(history, prompt)` variant (listed in
   step 001's deferred non-goals) would restore
   `/api/chat` semantics while still sharing the
   underlying machinery.

If you want the workspace-aware framing in a
script, build it yourself before calling
`llm_call`:

```mlpl
context = "Workspace shapes:\n  X: [100, 4]\n  loss: scalar"
question = "Why did the loss spike at step 42?"
prompt = context + "\n\nQuestion: " + question
reply = llm_call(host, prompt, "llama3.2")
```

## Why no web / WASM support

The browser sandbox cannot reach a localhost Ollama
server because of CORS. Setting `OLLAMA_ORIGINS` on
the Ollama side helps but breaks isolation; for
production-shaped deployments the right path is the
CLI server's proxy (Saga 21, `mlpl-serve`). When
that ships, the same `llm_call` source runs in the
web REPL: the WASM client POSTs the program to
`mlpl-serve`, the server runs the builtin natively
against a server-side allow-listed Ollama, and the
reply comes back through the same WebSocket. No
language-surface change is needed.

For now: no web-REPL lesson for `llm_call`. The
feature is CLI-only and a browser user would just
get a broken demo.

## Non-goals (deferred)

Listed in the contract; flagged here so users
landing on this doc see them at a glance:

- **Streaming SSE.** Single POST, whole-reply
  return. A future `llm_stream(...)` variant or
  callback-style argument can land later.
- **OpenAI-style tool calling / function-call
  schemas.** `tools` + `tool_choice` request
  fields plus the matching JSON-schema parser.
  Deferred until a concrete use case surfaces.
- **Multi-turn chat threading.** `llm_call` is
  single-turn. A `llm_chat(history_array,
  prompt)` variant could land later if useful;
  the REPL's `:ask` will pick it up first.
- **Request batching.** One call per builtin
  invocation. Easy to add if rate-limit pressure
  appears.
- **In-source auth secrets.** No bearer-token /
  API-key argument. Secrets must come from
  process env vars or the CLI server's
  server-side allow-list, never from MLPL source.
- **Teacher-model distillation pipeline.** Uses
  `llm_call` to generate soft labels, then
  distills into a student via existing
  `cross_entropy` + `adam`. Separate saga once
  the builtin is stable and the dtype machinery
  is in place.
- **Web / WASM support.** See above; Saga 21's
  CLI server is the path.
