# Maturation plan: hardening sw-MLPL before wider feedback

Planning doc (forward-looking). Distilled from
`docs/sw-mlpl-direction-proactive-response-to-criticisms.txt`.
Its job: name the work that makes sw-MLPL ready to invite a
larger community's criticism -- and the prepared answers for
the criticisms worth wanting. Companion-repo context:
`companion-repos.md`.

## The two things to fix first

The direction brief and the maintainer both converge on the
same short list: **libraries, not keywords; better
explanations (better errors + AI-coder integration + the
ability to explain and suggest).** Everything below serves one
of those two.

## 1. Context-aware error messages that suggest fixes

MLPL already errors LOUDLY with structured `EvalError`s
(kind + message; caught hard errors surface as `{kind,
message}` records). The maturation step is making each error
CONTEXT-AWARE and ACTIONABLE, so a newcomer -- or a coding
agent -- can fix it without external help.

Targets, in rough priority:

- **Did-you-mean.** An unknown builtin or `u:` name suggests
  the nearest catalog / user-fn name by edit distance
  ("unknown function `sofmax` -- did you mean `softmax`?").
- **Shape errors that show the shapes.** A matmul / broadcast
  mismatch names both operand shapes and the axis that
  conflicts, not just "shape mismatch".
- **Name-kind confusion.** Using `disp M` where `disp(M)` was
  meant, or `add` where `:add` was meant, points at the three
  kinds of name (the REPL already hints this in places;
  generalize it into the error surface).
- **Arity with the signature.** A wrong-arity call prints the
  expected signature from the catalog.
- **A fix suggestion field.** Extend the structured error
  record with an optional `suggestion` (and, where safe, a
  concrete `fix` snippet) so both humans and `swml-explain` /
  MCP tools consume the same guidance.

This is a discrete saga ("error-messages"): it touches the
`EvalError` types and their formatters, is heavily TDD-able
(assert the message names the culprit and the suggestion), and
pays off immediately in the browser demo where there is no AI
to lean on.

## 2. Better AI-agent integration

The brief's strongest strategic claim: sw-MLPL's largest
contribution may be **a language legible to AI through rich
semantic tooling rather than source text** (see the shared
thread in `companion-repos.md`). The maturation work has three
layers:

- **Interfaces** -- sw-mlpl-lsp (human editors) and
  sw-mlpl-mcp (agents) expose the same compiler artifacts.
  Prerequisites already queued: a `--check` parse-only flag
  and a machine-readable builtin-catalog export.
- **Semantic tools, not prompt engineering** -- `explain_ast`,
  `explain_tensor` (shape / dtype / axes / producer /
  consumer), `explain_pipeline` (the dependency graph),
  `find_optimization` (candidate transforms). These return
  STRUCTURED data an agent reasons over; the AST, spans,
  purity, and trace surfaces they need mostly exist already.
- **A model-agnostic adapter** (`swml-ai`): one `Assistant`
  trait (`explain` / `optimize` / `debug` / `document` /
  `visualize` / `suggest`) with interchangeable backends
  (Anthropic, OpenAI, Ollama, llama.cpp, OpenRouter, ...).
  The compiler and language never know which model is behind
  it; only the adapter changes. Note the hosting reality: the
  free github.io demo cannot ship a paid AI backend, so the
  full "helpful IDE" experience is a DESKTOP/installed story
  (editor -> LSP -> compiler -> AI adapter -> the user's own
  model); the browser demo stays AI-free and leans on the
  better error messages from section 1.

Deeper INTROSPECTION is the through-line: `tests()` /
`test_info` / `annotations` / `repr` / `describe` already
exist; the next tier is structured access to ASTs, typed IR,
shapes, purity, dependency graphs, tensor provenance, and
execution traces -- the raw material every tool above consumes.

## 3. Prepared answers to the criticisms worth wanting

These belong in the README/landing copy before wider posting,
phrased to keep the audience rather than fight it:

- **"Why not Python / NumPy / JAX / Mojo?"** Python is the
  current standard; sw-MLPL explores an alternative design
  space -- easier to optimize, compile, parallelize, and
  reason about, with native Rust integration and an
  educational focus. Not "Python sucks."
- **"Isn't this just APL / BQN / J / TensorFlow?"** It borrows
  from APL, BQN, Ramda, MLX, NumPy, and Rust, but is designed
  around READABILITY, compiler optimization, and ML
  experimentation rather than terseness or math notation.
- **"Show me something impossible elsewhere."** Lead with a
  demo, not a manifesto -- "attention in N readable lines,"
  the combinator/Y-combinator demos, the array-composition
  pervasion finale. The web playground is the answer.
- **"Performance?"** Honest: "Not yet -- the goal is compiler
  optimizations hard to get in Python while keeping code
  readable." Sets expectations correctly; "eventually" does
  not.
- **"Another DSL / Rust already exists / just use Python."**
  Calm, short: an exploration of whether ML-specific
  abstractions can improve readability AND optimization; not a
  Rust replacement (it generates/interops with Rust); an
  alternative design space, not a claim of superiority.

## 4. Market in layers (do not overwhelm)

The project currently tends to explain everything it may
BECOME; a new visitor needs what it IS today, in ~15 seconds:

> sw-MLPL is an experimental, Rust-native array programming
> language for machine learning -- inspired by the APL family
> but with readable keyword syntax, interactive
> experimentation, visualization, and native compilation.

Everything in that sentence is demonstrable now. The roadmap
and the semantic-AI vision are Layer 3 material, kept out of
the first screen. This is an editorial pass over README /
landing, not code -- but it gates a wider announcement.

## 5. Backend-independent IR (the enabler)

Flagged in the brief as the most important ARCHITECTURAL work,
even though users never see it: a backend-independent
intermediate representation between MLPL and its targets
(today Rust; tomorrow more). Everything in section 2's
semantic tooling (explain / optimize / find-optimization) and
the multi-backend roadmap depends on having an IR to talk
about. Large, foundational, and worth sequencing deliberately
rather than rushing -- its own program, not a quick step.

## Readiness checklist (before a wider call for feedback)

- [ ] Error messages suggest fixes for the top handful of
      newcomer mistakes (section 1).
- [ ] README Layer 1 + Layer 2 copy lands; roadmap demoted to
      Layer 3 (section 4).
- [ ] A "show something impossible" demo is the first thing a
      visitor sees.
- [ ] Prepared criticism answers are written down (section 3).
- [ ] `--check` + catalog export shipped (unblock LSP/MCP).
- [ ] The two-kinds-of-learning framing is stated
      (`companion-sw-mlpl-libraries.md`).

The semantic-AI interfaces (LSP/MCP/swml-ai) and the
backend-IR are maturation DIRECTIONS, not gates for the first
wider call -- but the error messages, the layered copy, and
the lead demo are.
