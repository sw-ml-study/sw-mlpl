# Plan: Demonstrating Speculative Decoding with MLX

Status: PLAN ONLY. Nothing installed or downloaded yet. This is written to
run on a disk-rich Mac (the current dev Mac has ~17 GB free; the model pairs
fit, but the bigger-gap variants want more headroom).

Grounded in `docs/speculative-decoding.txt` (a prior ChatGPT analysis) and a
June 2026 web check of the current `mlx-lm` state. Where the two disagree,
this plan notes it (see the Qwen3 caveat).

## 1. What we are demonstrating, and the honest win condition

Speculative decoding runs two models: a small **draft** model proposes a few
tokens; the large **target** model verifies them all in one forward pass; the
longest correct prefix is accepted and the first mismatch is resampled from
the target. The accepted text is drawn from the **target model's exact
distribution**.

So the framing "better than either model (speed AND accuracy)" is not what
this technique does. The correct, demonstrable win is:

| Claim                          | True? | Why                                          |
| ------------------------------ | ----- | -------------------------------------------- |
| quality ~= target              | yes   | output is the target's distribution          |
| quality > draft                | yes   | the target verifies/repairs the draft        |
| speed > target-alone           | yes   | many tokens verified per target pass         |
| accuracy > target              | NO    | it preserves, never exceeds, target behavior |
| speed > draft-alone            | NO    | the target still runs every step             |

The honest story to show: **you no longer choose between fast-but-dumb (small)
and smart-but-slow (medium) -- speculative decoding gives the medium model's
answer, faster.**

## 2. Why `mlx-lm` (and not our existing MLX, and not Ollama)

- **Our MLX is `mlx-rs` (Rust), compiled into `mlpl-serve --features mlx`.**
  It runs MLPL's *own* hand-built tiny transformers via `device("mlx")` (the
  MLX LoRA demo). It has no pretrained-LLM loader, no Qwen/Llama tokenizer,
  and no draft/verify loop. Building speculative decoding into it would mean
  writing model loading + tokenizers + KV-cache + the verify loop -- a large
  effort, not worth it just to measure the idea.
- **Ollama models are GGUF, not MLX weights** (`docs/speculative-decoding.txt`).
  Ollama names are only useful for picking a *family*.
- **`mlx-lm` is a separate Python package** on top of the same Apple MLX. It
  has speculative decoding built in (`--draft-model`, `--num-draft-tokens`,
  default 3) and loads ready-made MLX weights from the `mlx-community` org on
  Hugging Face.

So "install more" = the Python `mlx` + `mlx-lm` packages (small; this Mac has
neither -- both import-fail today). The big cost is the model weights.

## 3. Tools to install (on the disk-rich Mac)

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -U mlx mlx-lm huggingface_hub rich pandas
brew install hyperfine jq
```

- `mlx-lm` -- inference + speculative decoding + the `mlx_lm.generate` CLI.
- `hyperfine` -- clean wall-clock benchmarking with warmup + repeated runs.
- `rich` / `pandas` -- the comparison table in the bench script.

## 4. Model pairings (same family, same tokenizer, 4-bit `mlx-community`)

The draft and target MUST share the same tokenizer/vocabulary, so both come
from one family.

- **Lead demo -- Qwen2.5-Coder (best acceptance):**
  - draft: `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit` (~0.3 GB)
  - target: `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` (~4.3 GB)
  - Code/refactoring prompts have high draft-acceptance, so the speedup is
    most visible here. Project-relevant (Rust/Yew prompts).
- **Small-Mac fallback -- Llama 3.2:**
  - draft: `mlx-community/Llama-3.2-1B-Instruct-4bit` (~0.7 GB)
  - target: `mlx-community/Llama-3.2-3B-Instruct-4bit` (~1.8 GB)
  - Smaller speedup -- 3B is not much bigger than 1B (per the grounding doc).
- **Wider gap (needs the disk-rich Mac):** 0.5B draft -> 14B (~8 GB) or 32B
  (~18 GB) target. Bigger quality jump for "small alone" to climb, bigger
  potential speedup.

### Caveats on model choice

- **Qwen3 is risky.** The grounding doc suggests a Qwen3 0.6B -> 4B pair, but
  mlx-lm issue #846 reports Qwen3 speculative decoding **skips tokens /
  produces incorrect output**. Prefer **Qwen2.5 / Qwen2.5-Coder** or
  **Llama-3.x** until that is fixed; if Qwen3 is used, verify the
  greedy-identity check (Section 5) passes first.
- **Avoid MoE targets.** mlx-lm issue #1132: speculative decoding can be
  *slower* than target-alone on Mixture-of-Experts models whose active
  parameter count is near the draft size.
- The speculative `_step` itself has had performance edge cases (mlx-lm #250)
  -- which is exactly why we MEASURE rather than assume a win.

### Disk budget

| Item                         | Size approx |
| ---------------------------- | ----------- |
| python `mlx` + `mlx-lm` etc. | ~0.3 GB     |
| Qwen2.5-Coder-0.5B-4bit      | ~0.3 GB     |
| Qwen2.5-Coder-7B-4bit        | ~4.3 GB     |
| **Lead-demo total**          | **~5 GB**   |
| + 14B target instead of 7B   | +~8 GB      |
| + 32B target instead of 7B   | +~18 GB     |

## 5. Experiment design

Three configurations, the same prompt set, fixed seed, fixed `max_tokens`:

1. **Small alone** -- draft model as a standalone generator.
2. **Medium alone** -- target model standalone (the quality/speed baseline).
3. **Speculative** -- draft + target via `--draft-model`.

### Metrics

- **Decode speed:** tokens/sec (mlx-lm prints this with `verbose`/CLI) and
  wall-clock via `hyperfine --warmup 1 --runs 5`. Always discard a warmup run
  -- MLX lazy-compiles and loads weights on the first call.
- **Acceptance rate (speculative only):** fraction of draft tokens the target
  accepts. This is the lever that drives the speedup; report it per run. (Via
  the Python API / generation stats; CLI may need a small wrapper.)
- **Quality, two levels:**
  1. **Greedy-identity proof (the headline).** At `--temp 0.0`, speculative
     decoding is mathematically identical to the target. Generate each prompt
     with **medium-alone (greedy)** and **speculative (greedy)** and assert the
     outputs are **byte-identical**, while **small-alone** is visibly
     different/worse. This proves "speculative == the medium model's exact
     answer, produced faster" with no benchmark needed.
  2. **Task accuracy.** ~30 checkable items (e.g. small arithmetic/GSM8K-style
     questions, or factual Q&A with regex-checkable answers). Score % correct:
     expect small << medium ~= speculative. Quantifies the quality the speed
     buys back.
- **`num_draft_tokens` sweep:** 2, 3, 4, 6, 8. Too few underuses the draft;
  too many wastes target compute on rejected tokens. Plot speedup vs the
  setting to find the optimum and SHOW the tradeoff.

### Controls

- `--temp 0.0` (greedy) for the identity proof; a fixed temp + seed for any
  sampled runs.
- Fixed `--max-tokens 512` or more (short outputs hide the speedup).
- **Prompt type matters:** coding / boilerplate / structured-output prompts
  have high acceptance and win clearly; short creative prompts lower
  acceptance and hide the gain (grounding doc).
- Separate prefill (prompt processing) from decode; the speculative win is in
  decode.

### Prompt set (project-flavored, high-acceptance)

1. "Write a Rust function that parses a comma-separated list of integers with
   good error messages."
2. "Refactor this Rust parser into small pure functions with tests."
3. "Convert this TypeScript interface into Rust structs with serde
   annotations and unit tests."
4. "Write a Yew component skeleton with state, messages, update, view, and
   tests."
5. A couple of GSM8K-style arithmetic word problems (for the accuracy set).

## 6. Commands (verified against `mlx-lm`)

Baseline (target alone):

```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --prompt "Write a Rust function that parses a comma-separated list of integers with good error messages." \
  --max-tokens 512 --temp 0.0
```

Speculative:

```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --draft-model mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit \
  --num-draft-tokens 4 \
  --prompt "Write a Rust function that parses a comma-separated list of integers with good error messages." \
  --max-tokens 512 --temp 0.0
```

Wall-clock A/B with hyperfine:

```bash
hyperfine --warmup 1 --runs 5 \
  'mlx_lm.generate --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --prompt "Refactor this Rust parser into small pure functions with tests." --max-tokens 512 --temp 0.0' \
  'mlx_lm.generate --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --draft-model mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit --num-draft-tokens 4 --prompt "Refactor this Rust parser into small pure functions with tests." --max-tokens 512 --temp 0.0'
```

## 7. Deliverable

A single `spec_decode_bench.py` (Python, uses the `mlx_lm` API) that:

1. Loads the draft and target once.
2. Runs the prompt set in all three configs.
3. Records decode tok/s, wall-clock, and (speculative) acceptance rate.
4. Runs the greedy-identity check (assert speculative == target, show small
   differs).
5. Runs the task-accuracy set and scores each config.
6. Sweeps `num_draft_tokens`.
7. Prints a `rich`/`pandas` comparison table + a one-line verdict.

Location: a throwaway `experiments/spec-decode/` dir, or outside the repo --
it is Python + external model weights, not MLPL source, so it should not live
inside the Rust crates. Decide at write time.

## 8. Expected result table (what "value" looks like)

| Config            | decode tok/s | wall-clock | task acc | notes                       |
| ----------------- | ------------ | ---------- | -------- | --------------------------- |
| small alone (0.5B)| highest      | lowest     | low      | fast, wrong more often      |
| medium alone (7B) | low          | high       | high     | the quality baseline        |
| speculative       | mid-high     | mid-low    | = medium | medium quality, faster      |

Plus: at temp 0, speculative output is byte-identical to medium; acceptance
rate (e.g. 60-80% on code) explains the speedup; the `num_draft_tokens` sweep
shows a clear optimum.

## 9. Run order on the disk-rich Mac

1. `uv venv` + installs (Section 3).
2. Pull the Qwen2.5-Coder pair (~5 GB).
3. Baseline target run -- record tok/s.
4. Speculative run -- record tok/s + acceptance rate.
5. Greedy-identity diff (medium vs speculative -- must match).
6. Accuracy set across all three.
7. `num_draft_tokens` sweep.
8. Emit the table + verdict.

## Sources

- `docs/speculative-decoding.txt` -- prior ChatGPT analysis (grounding).
- mlx-lm `SERVER.md` -- `--draft-model` / `--num-draft-tokens`.
- mlx-lm issue #846 -- Qwen3 speculative decoding skips tokens (the revision
  to the grounding doc's Qwen3 recommendation).
- mlx-lm issue #1132 -- speculative decoding can hurt on MoE models.
- mlx-lm issue #250 -- speculative `_step` performance edge cases.
- LM Studio speculative-decoding docs -- mechanism + same-vocabulary
  requirement.
