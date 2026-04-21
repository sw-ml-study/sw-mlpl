# Using MLPL with Ollama (and other LLM servers)

> **Status:** planned -- Saga 19 (`v0.16` target). Not yet
> shipped in full. One preview command (`:ask <question>` in
> the CLI REPL) ships today -- see "Shipped today: `:ask`
> preview" below. Treat the rest of this doc as design, not
> reference.

## Shipped today: `:ask` preview (v0.11.x)

The CLI REPL has a `:ask <question>` command that POSTs to a
local Ollama server with the current workspace summary as
grounding context. It is a preview of Saga 19's REST
integration scoped to exactly one use case: "explain this to
me" in the middle of a session.

### Setup

```bash
# One-time: start Ollama and pull a model that fits on your
# machine. llama3.2 (2B) is the MLPL default because it loads
# fast and is small enough for laptops without a discrete GPU.
ollama serve &
ollama pull llama3.2

# In the MLPL CLI REPL
mlpl-repl
mlpl> v = iota(5) + [10, 20, 30, 40, 50]
mlpl> m = reshape(iota(12), [3, 4])
mlpl> :ask what's in my session?
```

The `:ask` command sends:

1. A short system prompt telling the model it is inside MLPL
   and should explain concisely (under ~200 words, plain
   prose, no made-up builtins).
2. The current `_demo` string if one is bound (web-REPL demos
   bind this automatically; CLI users can set it themselves
   with `_demo = "..."` to get the same grounding).
3. A compact workspace summary (variable names + shapes +
   `[param]` tags for trainables), capped at 40 entries so
   large sessions do not overflow the model's context.
4. The user's question.

### Configuration

- `OLLAMA_HOST` -- default `http://localhost:11434`. Change to
  point at a remote or containerized Ollama.
- `OLLAMA_MODEL` -- default `llama3.2`. Any model you have
  pulled locally works; `qwen2.5-coder` is a strong alternative
  when the question is code-shaped.
- Timeout is 120 seconds. Streaming is a Saga 19 follow-up; for
  now the REPL blocks until Ollama returns the full response.

### Why CLI-only, why not in the web REPL

Two reasons:

1. **CORS.** The browser's `fetch` to `http://localhost:11434`
   is blocked by Ollama's default CORS policy. Enabling it
   (`OLLAMA_ORIGINS="*" ollama serve`) works but is a sharp
   edge to put on the landing-page-accessible web demo.
2. **Scope.** `:ask` is most useful when you are mid-session
   and want a second opinion. The CLI is where that happens;
   the web REPL is the on-ramp for new users and stays
   dependency-free by design.

Web-REPL integration lands with Saga 19 proper, behind a
dedicated opt-in panel and a clearer "this talks to an
external service" warning.

### What the command is and is not

Is: a preview of a single Saga 19 surface, useful for
"describe this demo" / "what does :vars look like to an
outsider" / "what am I about to run" questions.

Is not: a REPL shell built on the LLM (you type MLPL, not
natural language), a codegen helper (you get prose answers,
not executable MLPL), or a proper REST client surface for use
inside MLPL programs. All three are Saga 19.

## Why this exists

MLPL's primary identity is "build your own tiny model" -- the
Saga 13 Tiny LM ran on CPU, fits in a handful of lines, and
trains reproducibly. Saga 19 adds the *complementary* ability to
call an external, larger LLM as a tool: for distillation, for
codegen helpers, for the "LLM as oracle" workflows that are
easier when you don't have to host a 70B model yourself.

Ollama is the default target because:

- It's free, local, runs on the same hardware as MLPL, and has a
  stable HTTP API.
- Model selection is a one-word dropdown (`llama3`, `qwen`,
  `phi3`, etc.); users don't need accounts or API keys to try it.
- The HTTP surface is boring (no SDK required), which keeps
  MLPL's dependency tree small.

`docs/tools.md` already mentions Ollama as the default for some
Softwarewrighter workflow tooling, so the convention is
consistent across the project.

## Saga 19 shape

From `docs/plan.md`:

> **Saga 19 -- LLM-as-tool integration.** REST client built-ins,
> teacher-model distillation workflows, codegen helpers.
> Intentionally last: secondary to the "build your own model"
> story.

The three pieces, roughly:

1. **REST client built-ins in `mlpl-runtime`.** A narrow set of
   primitives behind a server URL -- `llm_complete`,
   `llm_chat`, `llm_embed` -- with provider adapters so the same
   MLPL source works against Ollama, llama.cpp's server, an
   OpenAI-compatible endpoint, etc. Interpreter-only; the
   compile path stays closed (no `exec(string)`, no dynamic
   code loading).
2. **Teacher-model distillation.** The `experiment "..." { ... }`
   surface records the teacher's outputs alongside the student's
   metrics, so a distillation run is reproducible from the
   recorded data plus a seed.
3. **Codegen helpers.** Small, pure functions on top of
   `llm_complete` that format prompts for common cases -- "given
   these input/output pairs, write the MLPL expression that
   maps inputs to outputs". Optional; mostly a demo hook.

Web REPL caveat (see "CLI-only" section below): because LLM
calls go over HTTP and may hit arbitrary origins, this surface
is terminal-REPL-only. The web REPL will surface a clear
`Unsupported` error with a pointer at `mlpl-repl`.

## Intended API (not shipped)

### One-shot completion

```mlpl
# Point at a local Ollama, phi3 model
server = llm_server("http://localhost:11434", "phi3:3.8b")

reply = llm_complete(server,
                     "Give me three adjectives for a dog.",
                     0.2)                # temperature

# reply is a Value::Str; render inline or feed to tokenize_bytes
reply
```

### Chat-style conversation

```mlpl
msgs = [
  ["system", "You are a terse MLPL tutor."],
  ["user",   "How do I make a 3x4 matrix of zeros?"]
]
answer = llm_chat(server, msgs, 0.1)
answer                                  # -> "zeros([3, 4])"
```

Arguments are plain MLPL: strings, float temperatures, 2D string
arrays for chat messages. No SDK types leak into the language.

### Embeddings

```mlpl
texts = ["the cat sat", "on the mat", "dogs are loud"]
vecs : [n, d] = llm_embed(server, texts)
:describe vecs
# -> vecs -- array
#      shape: [n=3, d=384]
```

Embedding output is a standard `DenseArray` with labeled axes,
so downstream analysis (cosine similarity, k-means,
`scatter_labeled` after a PCA) is just normal MLPL -- no new
surface.

### Provider adapters

```mlpl
ollama  = llm_server("http://localhost:11434",        "qwen:4b")
llamacpp= llm_server("http://localhost:8080",         "default")
openai  = llm_server("https://api.openai.com/v1",     "gpt-4o-mini")
```

The server URL + model name pair is the provider abstraction.
Authentication tokens come from environment variables
(`OPENAI_API_KEY` etc.) not the MLPL source -- that keeps
secrets out of saved scripts and tutorial lessons. Ollama needs
no auth; that's why it's the default in examples.

### Distillation workflow

```mlpl
# Record a teacher's outputs on a small corpus
corpus  = load("data/prompts.txt")
teacher = llm_server("http://localhost:11434", "qwen:14b")

experiment "distill_qwen_to_tiny" {
  targets = [llm_complete(teacher, row, 0.0) for row in corpus]

  # Student: MLPL Tiny LM from Saga 13, trained to match targets
  student = chain(embed(V, d, 0), causal_attention(d, h, 1),
                  rms_norm(d), linear(d, V, 2))
  train 500 {
    adam(cross_entropy(apply(student, X), targets_tokens),
         student, 0.001, 0.9, 0.999, 0.00000001);
    loss_metric = cross_entropy(apply(student, X), targets_tokens)
  }
}
```

The experiment block captures the teacher's server URL and model
name alongside the student's final loss, so the run is
reproducible given the same seed + a reachable teacher.

## Running Ollama for MLPL

Ollama is an out-of-band install (nothing in this repo pulls it
in). Once the MLPL side lands:

```bash
# one-time install on macOS
brew install ollama

# pull a small model
ollama pull phi3:3.8b

# start the server (default port 11434)
ollama serve

# in another shell, in MLPL:
cargo run -p mlpl-repl
#   mlpl> server = llm_server("http://localhost:11434", "phi3:3.8b")
#   mlpl> llm_complete(server, "hello", 0.2)
```

llama.cpp's `server` binary presents the same OpenAI-compatible
HTTP shape, so point at `http://localhost:8080` instead.

## CLI-only by design

Network calls from the web REPL (Yew/WASM) are deliberately out
of scope. Three reasons:

1. **CORS.** The browser won't let the WASM bundle call
   arbitrary origins. Working around this by proxying through a
   server defeats the "the live demo needs no server" property.
2. **Secret hygiene.** If the web REPL called `/v1/chat/...`
   with an API key typed into the page, that key would ride
   every refresh and be trivially exfiltrated by anyone who
   could load the page with a different MLPL expression.
3. **Demo integrity.** A demo that works on the live REPL at
   <https://sw-ml-study.github.io/sw-mlpl/> should work *without
   a running LLM on your machine*. Every web demo today is
   pure-client; adding one that needs an external server
   silently broken for the majority of visitors is a regression.

Saga 19's built-ins therefore return
`EvalError::Unsupported("llm_complete: ... CLI REPL only,
point at mlpl-repl")` in the web REPL. The "Demo collection" in
`apps/mlpl-web/src/demos.rs` must not include LLM-backed
entries. See the "Web vs CLI demos" section in
`docs/repl-guide.md` (landing alongside Saga 19) for the rule.

## What you can do today

- **Tokenize text with `tokenize_bytes` / `train_bpe`.** The
  whole Saga 12 tokenizer surface is already in place, and it's
  what a future `llm_complete` would feed to the student model
  in a distillation loop.
- **Build a tiny student model.** The Saga 13 Tiny LM demo and
  tutorial lessons are the student-model template. When
  Saga 19's teacher-call lands, the student side is the code you
  already have.
- **Track runs with `experiment "..." { ... }`.** The
  distillation workflow above leans on this existing surface.
  Practice with it now -- Saga 12's "Experiments" tutorial
  lesson walks through it.

## Related

- `docs/plan.md` -- Saga 19 entry (and why it's deliberately
  last)
- `docs/saga.md` -- live saga status
- `docs/tools.md` -- the repo's existing Ollama convention
- `docs/using-mlx.md` -- the MLX backend (Saga 14), which is
  what you'd want for *training* the student model once it
  ships
- `docs/repl-guide.md` -- the REPL commands + CLI vs web
  surface rules
