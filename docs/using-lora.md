# Using MLPL for LoRA Fine-Tuning

> **Status:** reference. Shipped in Saga 15 (v0.13.0). The
> design sketch that motivated this saga is the
> `docs/plan.md` Saga 15 entry; this doc is the
> shipped-surface reference + honest retrospective.

## What LoRA is

Low-Rank Adaptation (LoRA) is a parameter-efficient
fine-tuning technique: given a trained base model, instead
of updating every weight during fine-tune, freeze the base
and train two small low-rank matrices `A` and `B` per
linear layer. The effective weight becomes `W + (alpha /
rank) * A @ B` where `A` is `[in, rank]`, `B` is
`[rank, out]`, and `rank` is typically 4-64 (vs the
`in * out` full-matrix size). For a Tiny LM with a
`[32, 128]` linear, rank-8 LoRA adds `32*8 + 8*128 = 1280`
trainable params vs the full `32*128 = 4096`. At real LLM
scale the savings are dramatic: a 7B-param Llama trained
with LoRA typically ships ~10M trainable adapters.

The MLPL surface makes this a three-builtin workflow:

```mlpl
base = chain(...)             # pre-trained model
student = lora(base, 8, 16.0, 0)
train N { adam(loss, student, ...); loss_metric = loss }
# -> every base param frozen; only adapters train.
```

## The surface (three builtins + one variant)

### `freeze(m) -> scalar 0`

Marks every name in `m.params()` as frozen in
`env.frozen_params`. `adam` and `momentum_sgd` skip frozen
names at the optimizer-update stage. Gradient computation
is unchanged -- the chain rule still flows through frozen
params; only the *update* is suppressed.

- Bare model identifier (looked up in `env.models`) or any
  expression that evaluates to `Value::Model`.
- Idempotent; a repeated `freeze(m)` is a no-op.
- Contract: `contracts/eval-contract/freeze.md`.

### `unfreeze(m) -> scalar 0`

Inverse of `freeze`. Removes every `m.params()` name from
`env.frozen_params`. Useful after `lora()` to reopen the
base to training (see the auto-freeze note below).

### `lora(m, rank, alpha, seed) -> Model`

The workhorse. Clones `m`'s spec tree, replaces every
`Linear` node with a `LinearLora` that owns the cloned
base `W`, `b` plus two fresh adapter matrices `A`, `B`,
and auto-freezes every non-adapter parameter in the
returned student. Contract:
`contracts/eval-contract/lora.md`.

- `rank`: positive integer, `0 < rank <= min(in, out)` for
  every wrapped `Linear`.
- `alpha`: LoRA scaling; the forward uses `alpha / rank`
  as the multiplier on the adapter delta. Literature
  convention is `alpha = 2 * rank` (so scale = 2); the
  demo uses `alpha = 16, rank = 8`.
- `seed`: each replaced `Linear` uses PRNG seed
  `seed + i` (i = running index over wrapped linears) for
  its A init, so same-shape adapters in one call get
  independent deltas and two `lora` calls with the same
  seed produce bit-identical `A` matrices.

### `ModelSpec::LinearLora`

New variant on the `ModelSpec` enum; owns `w, b, a,
b_adapter, in_dim, out_dim, rank, alpha`. Forward (in
both CPU `apply_model` and the autograd tape):

```
y = X @ W + (alpha / rank) * X @ A @ B + b
```

Bias is broadcast the same way `ModelSpec::Linear` does
it (`ones([n, 1]) @ b[1, out]`). The autograd tape
composes through the same matmul / scalar-mul / add
primitives it already uses for bare `Linear`, so no new
backward formulas were needed.

## Initialization

- **`A`** -- `randn(seed + i, [in, rank]) * (1 /
  sqrt(in))`. Scaled by `1/sqrt(in)` so `X @ A` has O(1)
  magnitude regardless of input dim (standard LoRA
  initialization).
- **`B`** -- `zeros([rank, out])`. Before any gradient
  step, the adapter delta is zero, so
  `apply(lora_m, X) == apply(m, X)` elementwise. This is
  the standard LoRA zero-init-B property; it means the
  wrapped model starts exactly at the base's behavior and
  only diverges as the adapters train.

Both invariants are pinned by
`crates/mlpl-eval/tests/lora_tests.rs` +
`crates/mlpl-eval/tests/lora_forward_tape_tests.rs`.

## Automatic freeze of the base

After rewriting the tree, `lora` walks the student's full
param list and marks every name that is NOT a new adapter
(`__lora_A_*` or `__lora_B_*`) as frozen. This includes:

- Cloned base `W`, `b` of every wrapped `Linear`.
- Embedding tables (`__embed_E_*`).
- Attention projections (`__attn_Wq_*`, `__attn_Wk_*`,
  `__attn_Wv_*`, `__attn_Wo_*`).
- `rms_norm` carries no parameters; no-op.

The design goal: the user writes `student = lora(base,
...)` and immediately gets the "frozen base, trainable
adapters" semantics -- no need to enumerate which base
params to freeze. If they want full fine-tuning, they call
`unfreeze(student)` after `lora()`, which clears the
frozen set for every student param (adapters stay
trainable because they were never frozen).

## Device propagation

The adapter matrices inherit the cloned base `W`'s device
tag. If the base was moved to MLX via
`to_device(base, "mlx")` before `lora()`, the student's
adapters land on MLX too. In practice the common pattern
is:

```mlpl
base = chain(...)                    # built on CPU
student = lora(base, 8, 16.0, 0)    # CPU-tagged student
device("mlx") {
  to_device(student, "mlx")          # walk student.params(), stamp mlx
  to_device(instr_X, "mlx")
  train 50 { adam(...) }             # MLX dispatch
}
```

## The shipped demos

### `demos/lora_finetune.mlpl` (CPU, any host)

1. Pre-train a Saga 13 Tiny LM (V=280, d=32, h=1) on the
   `tiny_shakespeare_snippet` preloaded corpus for 100
   Adam steps under
   `experiment "lora_base_pretrain" { }`.
2. Synthesize a short Q/A instruction corpus (9 Q/A
   pairs, tokenized with the same Shakespeare BPE).
3. `student = lora(base, 8, 16.0, 0)` -- clones +
   allocates rank-8 adapters + auto-freezes every
   non-adapter param.
4. Fine-tune 50 Adam steps under
   `experiment "lora_finetune" { }`. Base bit-identical
   throughout; adapters move.
5. `loss_curve(last_losses)` renders the fine-tune
   descent.

Manual run: final fine-tune cross-entropy ~2.18 on the
instruction corpus.

```bash
./target/release/mlpl-repl -f demos/lora_finetune.mlpl
```

### `demos/lora_finetune_mlx.mlpl` (Apple Silicon, CLI)

Mirror of the CPU demo with the fine-tune train block
wrapped in `device("mlx") { ... }` and a
`to_device(student, "mlx")` + `to_device(instr_X,
"mlx")` prologue. Base pre-training stays on CPU (cheap
+ deterministic; the MLX thesis at this scale is
source-parity, not speed -- see below).

```bash
cargo run -p mlpl-repl --features mlx --release -- \
  -f demos/lora_finetune_mlx.mlpl
```

### Web REPL tutorial lesson

The web REPL at
<https://sw-ml-study.github.io/sw-mlpl/> ships a "LoRA
Fine-Tuning" lesson that runs a browser-interactive
variant (V=8, d=4, rank=2) so the full forward and a 10-
step fine-tune render in a couple seconds in WASM. See
`apps/mlpl-web/src/lessons.rs`.

## Performance

Measured on an M-class laptop (`cargo bench -p
mlpl-bench --features mlx --bench mlx_vs_cpu --
lora_finetune_step`):

| Path | Cold | Warm |
|---|---:|---:|
| CPU | 208 us | 164 us |
| MLX | 1.45 ms | 1.11 ms |

MLX is **0.15x** of CPU on this workload -- a step down
from Saga 14's `tiny_lm_train_step` (0.26x) and Saga
20's `neural_thicket_variant_loop` (0.25x) at the same
Tiny LM scale. Cause: LoRA doubles the matmul count per
linear (`X @ W` AND `X @ A @ B` plus the scalar scale)
and the adapter matmuls (`[T, rank=2]` and
`[rank=2, out]`) are too small to amortize MLX's per-op
kernel-launch + f32 round-trip overhead. At d=512 the
ratio would flip. See `docs/benchmarks.md` for the
4-point bottleneck analysis (unchanged since Saga 14).

At **this** scale the MLX value is source-level parity,
not speed -- correctness is verified by
`crates/mlpl-eval/tests/lora_mlx_demo_tests.rs` (CPU and
MLX fine-tune losses + every student param agree within
fp32 tolerance; frozen base bit-identical on both paths,
confirming the optimizer's frozen filter is
backend-independent).

## Where this runs

The LoRA language surface is pure Rust; `freeze`,
`unfreeze`, `lora`, and `LinearLora` compile into every
environment the evaluator ships to:

| Environment | CPU LoRA | MLX LoRA |
|---|:---:|:---:|
| `mlpl-repl` CLI (native) | yes | yes (`--features mlx`) |
| Web REPL (WASM) | yes (slow at full scale) | no (WASM cannot link `mlx-rs`) |
| Emacs client | via CLI REPL | via `--features mlx` CLI |

Saga 21's planned CLI-server architecture (browser UI +
server-side `mlpl-serve` with MLX) is the eventual path
for "web UI + GPU-accelerated LoRA"; see
`docs/configurations.md` for the configuration matrix.

## Parity testing

- `crates/mlpl-eval/tests/lora_tests.rs` (14 tests):
  rewrite correctness -- param shapes, zero-init B,
  scaled-randn A, auto-freeze coverage (base W/b +
  embedding + attention), chain walking, disjoint clones,
  error paths (rank=0, rank > min(in,out), non-model,
  nested `lora`).
- `crates/mlpl-eval/tests/lora_forward_tape_tests.rs`
  (6 CPU + 1 MLX-gated): forward identity before training,
  hand-constructed formula, alpha/rank scaling, Adam
  isolation (W/b bit-identical after training, A/B move),
  unfreeze round-trip, chain composition, CPU-vs-MLX
  parity within 1e-3.
- `crates/mlpl-eval/tests/lora_finetune_tests.rs` (2
  CPU): end-to-end with a Tiny LM-shaped base + 3-step
  fine-tune; every non-adapter bit-identical, every
  adapter moved, source base binding untouched.
- `crates/mlpl-eval/tests/lora_mlx_demo_tests.rs`
  (3 MLX-gated): CPU vs MLX fine-tune parity on the
  losses curve AND every student param; frozen-base
  bit-identical on both paths; demo file parses.

## Not shipped (deferred follow-ups)

- **QLoRA / 4-bit quantization.** Quantization needs
  per-tensor scale/zero-point handling and its own parity
  harness. Deferred to a future saga; LoRA ships in
  fp64/fp32 today.
- **Selective layer attachment.** LoRA-in-the-wild
  commonly adapts only attention projections (Wq, Wv) or
  a subset of layers. Saga 15 ships the uniform "every
  Linear gets an adapter" variant; a `lora(m, rank,
  alpha, seed, layers: "attention_only")` variant
  composes naturally with Saga 20's family walker and is
  the obvious next step.
- **Adapter merging (`merge_lora(m)`).** Bake the
  adapter back into the frozen `W` for inference
  deployment. Nice-to-have once an inference path cares.
- **Multi-adapter composition.** One base, multiple
  swappable adapters, or adapter stacking. Follow-up
  after plain LoRA proves out.
- **Real pretrained checkpoints.** The demo fine-tunes a
  from-scratch Tiny LM. A Llama-class base needs Saga
  15+ checkpoint loading or Saga 19's LLM sidecar; both
  out of scope here.
- **Nested `lora()`.** Applying `lora` to an
  already-lora'd model is an explicit error today; the
  semantics (does it add more adapters on top, or
  replace?) is unresolved.

## Related

- `contracts/eval-contract/freeze.md` --
  `freeze` / `unfreeze` contract.
- `contracts/eval-contract/lora.md` -- `lora` + LinearLora
  contract.
- `demos/lora_finetune.mlpl` -- CPU demo.
- `demos/lora_finetune_mlx.mlpl` -- MLX variant.
- `docs/benchmarks.md` -- the Tiny-LM-scale MLX
  bottleneck analysis that explains the 0.15x ratio.
- `docs/using-mlx.md` -- MLX reference doc; the
  `device("...") { }` + `to_device` surface the MLX demo
  uses.
- `docs/configurations.md` -- CLI vs web vs CLI-server
  matrix; Saga 15 footnotes [4] and [5] cover the
  "where does MLX LoRA run" question.
- `docs/using-perturbation.md` -- sibling Saga 20 doc;
  same structure.
