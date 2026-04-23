#  Demo scripts

Step-by-step recipes for the demos you can show today. Each
demo names the artifact it exercises, one-line "what this
shows", and the literal commands (copy-paste-able) plus what to
watch for on screen.

This doc lives next to the shipped demos in `demos/` and the
shipped tutorial lessons in `apps/mlpl-web/src/tutorial.rs`;
pick the demo that matches your audience. The running
assumption is you are on Apple Silicon -- the MLX demos need
`--features mlx`, everything else runs anywhere.

## Setup (once)

```bash
cd ~/github/sw-ml-study/sw-mlpl
git pull                                    # latest main
cargo build -p mlpl-repl --release          # cold-build REPL
cargo build -p mlpl-repl --release --features mlx  # MLX-enabled REPL
```

The first MLX build is long (vendored `mlx-rs` + its C++/Metal
deps); subsequent builds are fast. The stable installed
`mlpl-repl` under `~/.local/softwarewrighter/bin/` is the
non-MLX CPU build -- use it for every non-MLX demo to skip the
`cargo run` warm-up.

Convenient aliases for this doc:

```bash
REPL_CPU=~/.local/softwarewrighter/bin/mlpl-repl
# MLX-enabled REPL lives under `target/release/` after
# `cargo build -p mlpl-repl --features mlx --release`. Per
# project policy `sw-install` is not run for in-progress
# builds, so this alias points at the target dir:
REPL_MLX="target/release/mlpl-repl"
```

## Demo 1 -- "Platform thesis in 10 lines" (2 min, CPU)

**What it shows.** A few lines of MLPL train a tiny transformer
language model end-to-end on a Shakespeare snippet, generate
40 tokens from a prompt, and render the attention pattern as an
SVG heatmap. The core claim of the platform (array-notation
syntax + first-class viz + autograd + optimizers) in one
artifact.

**Why.** This is the Saga 13 "does the whole thing actually
work" demo. No GPU, no external LLM.

**How.**
```bash
$REPL_CPU -f demos/tiny_lm_generate.mlpl
```
Runs ~10-15 seconds on an M-class laptop. Watch for:
- Loss values printed per training step (decreasing).
- Final `text_out` value: the generated continuation of the
  prompt "to be ".
- An SVG heatmap written to stdout that shows which earlier
  positions the causal attention focused on.

**Follow-up questions to expect.**
- "Is this a real pretrained model?" No, trained from scratch
  on a tiny corpus in seconds. Point to
  `demos/tiny_lm_generate.mlpl` itself as the source.
- "Can I see the attention heatmap in the browser?" Yes -- open
  `https://sw-ml-study.github.io/sw-mlpl/` and run the
  "Training and Generating" tutorial lesson.

## Demo 2 -- "Same source, MLX backend" (3 min, Apple Silicon)

**What it shows.** The same training loop wrapped in
`device("mlx") { ... }` dispatches matmul, softmax, autograd,
and Adam through Apple's MLX backend with zero source changes
otherwise. Source-level parity: same loss curve shape (within
fp32 tolerance), same shapes and labels, same `experiment`
record.

**Why.** Demonstrates the Saga 14 "one source, many backends"
thesis. Honest caveat: at Tiny-LM scale the MLX path is
currently slower than CPU (see Demo 5); the win here is the
portability, not the speed.

**How.**
```bash
$REPL_MLX -f demos/tiny_lm_mlx.mlpl
```
Runs tens of seconds. Watch for:
- Loss curve identical in shape to Demo 1 -- the numerical
  values match within ~1e-3.
- No "falling back to CPU" warning printed -- the MLX dispatch
  is actually exercising the GPU path.

**Follow-up questions to expect.**
- "How much faster is it?" At this scale, slower.
  `docs/benchmarks.md` has the measured numbers and a four-item
  bottleneck breakdown. At d=256+ inner dims (a real small LLM)
  the ratio would flip.
- "What happens if I don't pass `--features mlx`?" A one-time
  warning prints and the block transparently falls back to CPU.
  Correctness preserved; device scope becomes a no-op.

## Demo 3 -- "Web REPL tour" (5 min, browser)

**What it shows.** Every Saga 14 primitive surface, interactive,
in a browser: arithmetic -> arrays -> variables -> tensors ->
autograd -> optimizers -> Model DSL -> language-model training
-> running on MLX. No install required.

**Why.** The shortest path to "try this yourself." Also useful
for audiences who want to see the tutorial-driven on-ramp
rather than a single demo script.

**How.** Open `https://sw-ml-study.github.io/sw-mlpl/` and
click "Tutorial" in the top-right. 22 lessons, each with a
worked intro + runnable examples + a "try it" prompt. Suggested
flow for a demo:
1. Lesson 1 ("Hello Numbers") to show the REPL is live.
2. Skip to "Automatic differentiation" to show `grad(sum(w*w),
   w)` computing `2*w` on the fly.
3. Skip to "Language Model Basics" to show the Tiny LM forward.
4. Close with "Running on MLX" to show the device block is just
   a scoped form -- browser runs everything on CPU, MLX
   dispatch is the native-only path.

**Follow-up questions to expect.**
- "Is this really running in the browser?" Yes -- compiled to
  wasm32 via `trunk build --release` (see
  `scripts/build-pages.sh`). The entire interpreter runs
  client-side.
- "What about the SVGs?" They render inline in the REPL
  output; scroll up in the browser after running
  `svg(attn_w, "heatmap")`.

## Demo 4 -- "Parity test wall" (2 min, any host)

**What it shows.** The test suite that ran behind every Saga 14
step. Demonstrates that the MLX path is numerically correct --
every primitive, every model component, autograd, optimizer,
and the Tiny LM end-to-end all match CPU within documented
fp32 tolerance.

**Why.** Credibility check. The speed story has a caveat (Demo
5) but the correctness story is complete and testable.

**How (non-Apple host, no MLX dispatch):**
```bash
cargo test --workspace 2>&1 | tail -20
```
Hundreds of tests pass across every crate.

**How (Apple Silicon, MLX dispatch active):**
```bash
cargo test -p mlpl-eval --features mlx 2>&1 | tail -20
cargo test -p mlpl-mlx  --features mlx 2>&1 | tail -20
```
Watch for:
- `device_block_tests` (15 tests): parser + eval semantics of
  the device scoped form + nesting + MLX matmul parity.
- `device_dispatch_tests` (9 tests): Model DSL dispatch +
  to_device + cross-device error + Saga 11 model parity +
  end-to-end Tiny LM forward.
- `grad_mlx_tests` (16 tests): per-primitive gradcheck parity
  (add/mul/div/sub/neg/exp/log/tanh/sigmoid/relu/softmax/sum/
  mean/transpose/matmul) + chained ops + cross_entropy +
  apply(linear) integration.
- `train_mlx_tests` (5 tests): one-Adam-step parity,
  one-momentum-sgd-step parity, `train 3 { adam(...) }`
  last_losses parity, schedules under device(mlx),
  experiment loss_metric capture.
- `tiny_lm_mlx_demo_tests` (2 tests): demo-file parses,
  micro-variant (V=60, d=16, T=8, 3 steps) loss curve matches
  CPU within 1e-3.
- `parity_tests` in `mlpl-mlx` (27 tests): per-primitive CPU vs
  MLX values within fp32 tolerance.
- `tiny_lm_forward_tests`: structurally identical Tiny LM
  forward on both runtimes via a function-pointer backend
  table, final loss within 5e-4.

**Follow-up questions to expect.**
- "What tolerance?" Documented per-step in the test
  files -- typically 1e-5 for single-kernel ops, 1e-4 for
  autograd composition, 1e-3 for multi-layer chained forwards.

## Demo 5 -- "Honest benchmark" (3 min, Apple Silicon)

**What it shows.** A Criterion harness measures MLX vs CPU
wall clock on the same two workloads. Reports both cold (first
call, MLX compile overhead included) and warm (steady state)
timings. The saga plan set a 5x warm-path gate; we missed it.

**Why.** Honesty. The bottlenecks are understood and
documented; the next natural step is an "MLX throughput"
follow-up.

**How.**
```bash
cargo bench -p mlpl-bench --features mlx --bench mlx_vs_cpu 2>&1 | tail -40
```
Runs ~45s. Watch for two Criterion groups:
- `reshape_reduce_100x100`: CPU 68.5 us / MLX 81.1 us warm
  (MLX 0.84x slower).
- `tiny_lm_train_step`: CPU 619 us / MLX 2.36 ms warm (MLX
  0.26x slower).

Cold timings print first for each group (one-shot wall-clock
before Criterion's warm-up). They are 3-4x slower than the
warm numbers on MLX because of the first-call compile
overhead -- this is expected.

**Follow-up questions to expect.**
- "Why is it slower?" Four reasons, documented in
  `docs/benchmarks.md`: f32/f64 round-trip per op, no graph
  fusion (every op `.eval()`s immediately), step-006 tape
  re-materialization doubles forward work, Tiny LM inner dims
  (d=16/32) put kernel-launch overhead comparable to
  arithmetic.
- "When does MLX win?" At d=256+ or larger inner dims; the
  kernel-launch overhead becomes a rounding error and the
  matmul FLOPs dominate. Saga 14 deliberately shipped at Tiny
  LM scale; the perf follow-up targets the four bottlenecks.

## Demo 6 -- "Same source, compile-to-Rust" (2 min, any host)

**What it shows.** The same MLPL source that the interpreter
runs can be lowered to Rust via the `mlpl!` proc macro or the
`mlpl build` CLI, then compiled to native code or to WebAssembly.
Interpreter -> 9x measured speedup for typical workloads (see
`docs/benchmarks.md`). Complementary to Demo 2 (MLX is the
accelerator-backend story; compile-to-Rust is the
native-binary story).

**Why.** Shows the other half of "one source, many backends."
Useful for audiences interested in deployment, not just
exploration.

**How.**
```bash
cargo bench -p mlpl-bench --bench interp_vs_compiled 2>&1 | tail -30
```
Watch for each workload's `interp` vs `compiled` row; ratios
range from 1.6x (memory-traffic-bound `reshape_reduce_100x100`)
to 6.6x (small sweep `iota_reduce_100`).

**Follow-up questions to expect.**
- "Can I produce a standalone binary?" Yes: `cargo run -p
  mlpl-build -- my_program.mlpl -o my_program`. See
  `docs/compiling-mlpl.md`.
- "Can I cross-compile to WASM?" Yes: `cargo run -p mlpl-build
  -- --target wasm32-unknown-unknown my_program.mlpl -o
  my_program.wasm`.

## Demo 7 -- "Neural Thickets" (3 min, CPU)

**What it shows.** Take a trained base Tiny LM, perturb its
weights into 16 variants across 4 families (`all_layers`,
`attention_only`, `mlp_only`, `embed_and_head`), score each
on a held-out string, render a `[family x seed]`
specialization heatmap, pick the top-K best with
`argtop_k`, and average logits across all variants for an
ensemble cross-entropy. The MLPL-as-research-platform demo
in ~100 lines of source.

**Why.** Saga 20 surface in action. Shows that
`clone_model` + `perturb_params` + `argtop_k` + `scatter`
compose into a non-trivial weight-space-search algorithm
without leaving the language. Useful for audiences who care
about "can I use this for my own research question?"

**How.**
```bash
$REPL_CPU -f demos/neural_thicket.mlpl
```
Runs ~10-20 seconds on an M-class laptop. Watch for:
- Base training loss decreasing across 100 `train` steps
  inside the `experiment "thicket_base"` block.
- The final value (`ens_metric`) printed: cross-entropy of
  the 16-variant ensemble on the held-out string.
- `heat_svg` produced via `svg(heat, "heatmap")`: rows are
  perturbation families (top -> bottom: all_layers,
  attention_only, mlp_only, embed_and_head), columns are
  seeds.

**Follow-up questions to expect.**
- "Why not strict top-K ensembling?" The flat `best_idx` from
  `argtop_k` does not include family info; selecting the
  matching family string per index needs string-array
  indexing, which the Saga 20 surface does not yet have.
  CPU demo averages all 16 variants instead. Same forward
  composition; the strict top-K case is a follow-up once
  string-indexing or per-index dispatch lands.
- "Does it need CUDA?" No. CPU runs end-to-end. Step 005 of
  Saga 20 wraps the variant loop in `device("mlx")` for the
  Apple Silicon variant.
- "Where is the design sketch?"
  `docs/mlpl-for-neural-thickets.md` walks through the
  thicket idea, the four new builtins, and what is and is
  not in scope for Saga 20. `docs/using-perturbation.md`
  (step 006) is the user-facing retrospective.

## Demo 8 -- "LoRA fine-tune" (3 min, CPU)

**What it shows.** Pre-train a Saga 13 Tiny LM on the
Shakespeare snippet, wrap it with rank-8 LoRA adapters
(`lora(base, 8, 16.0, 0)`), and fine-tune the adapters on
a tiny synthetic instruction corpus. The base stays
bit-identical across the fine-tune; only the low-rank
adapter matrices `A [in, r]` and `B [r, out]` train. The
whole pipeline (base train + adapter wrap + fine-tune)
fits in ~40 lines of MLPL.

**Why.** Saga 15 surface in action. Shows that `freeze`,
`lora`, and the auto-freeze semantics ("everything in the
student tree except the new adapters is frozen at rewrite
time") compose with the existing Model DSL + adam + train
without any new training concept. Useful for audiences
who ask "can I fine-tune a base model in this language
the way I would in PyTorch + peft?"

**How.**
```bash
$REPL_CPU -f demos/lora_finetune.mlpl
```
Runs ~10-15 seconds on an M-class laptop. Watch for:
- Two `experiment "..." { train N { ... } }` blocks: first
  `lora_base_pretrain` moves the base, then
  `lora_finetune` moves only the adapters.
- Final `loss_metric` printed for each block: base pretrain
  settles around the Shakespeare corpus's entropy floor;
  the fine-tune final loss is around 2.18 on the
  instruction corpus (loose reproducibility bound -- the
  exact value shifts with seeds and tokenizer training
  order).
- `loss_curve(last_losses)` at the end writes an SVG of
  the fine-tune loss descent.

**Follow-up questions to expect.**
- "Does the base move?" No. `lora()` auto-freezes every
  non-adapter param in the student (cloned base W/b,
  embedding table, attention projections). The integration
  test `crates/mlpl-eval/tests/lora_finetune_tests.rs`
  asserts this bit-identically.
- "MLX variant?" `demos/lora_finetune_mlx.mlpl` (Saga 15
  step 005) wraps the fine-tune loop in `device("mlx") { }`.
  Apple Silicon CLI only; see `docs/configurations.md` for
  the CLI-vs-web matrix.
- "Can I train everything, base and adapters together?"
  `unfreeze(student)` after `lora()` reopens every frozen
  param. Unusual for LoRA but legal.
- "What about QLoRA / 4-bit quantization?" Deferred to a
  future saga. See `docs/using-lora.md` (Saga 15 step 006)
  for the full deferred-follow-up list.

## Quick-reference: what each demo proves

| Demo | Proves | Time | Host |
|---|---|---:|---|
| 1. Tiny LM CPU | Platform thesis end-to-end | 2 min | any |
| 2. Tiny LM MLX | One source, many backends | 3 min | Apple Silicon |
| 3. Web REPL tour | Zero-install on-ramp | 5 min | browser |
| 4. Parity test wall | Correctness credibility | 2 min | any |
| 5. MLX benchmark | Honest performance status | 3 min | Apple Silicon |
| 6. Compile-to-Rust | Native / WASM deployment | 2 min | any |
| 7. Neural Thickets | Weight-perturbation specialization | 3 min | CPU |
| 8. LoRA fine-tune | Frozen base + trainable adapters | 3 min | CPU |

## Suggested demo orders

**Short (5 min).** Demo 1 alone. Platform thesis, done.

**Technical audience (15 min).** Demo 1 -> Demo 4 -> Demo 2 ->
Demo 5. Correctness first, then the interesting architecture
conversation.

**Research-curious audience (12 min).** Demo 1 -> Demo 3 ->
Demo 7. End-to-end thesis, hands-on browser tour, forward-
looking roadmap.

**Deployment-curious audience (9 min).** Demo 1 -> Demo 6 ->
Demo 2. What it does, how it ships, where it is going.

## Related

- `demos/` -- the .mlpl source files these scripts reference
- `apps/mlpl-web/src/tutorial.rs` -- the web REPL lessons
- `docs/benchmarks.md` -- measured numbers for Demos 5 and 6
- `docs/using-mlx.md` -- shipped-API reference for Demo 2
- `docs/mlpl-for-neural-thickets.md` -- the Demo 7 sketch
- `docs/compiling-mlpl.md` -- the Demo 6 reference
