# Future saga: Stable Diffusion (text-to-image) in sw-MLPL

**Status (2026-07-25): PARKED after step 001.** The saga kicked off in
June (step 001 diffusion-2d-demo shipped: linspace/running_product, the 2D
diffusion demo, glossary terms); step 002 (conv2d) was never started and
the saga is archived at `.agentrail-archive/stable-diffusion-20260725T084701/`.
Resume from step 2 below after the connect-telemetry saga
(`docs/saga-connect-telemetry.md`). Seed input for
`agentrail init` (name, vision, phased steps, what it builds on, the
browser-vs-connect split, risks). Companion to
`docs/future-sagas-multibackend-specdecode.md` and
`docs/gpu-demos-roadmap.md`; this one is prioritized ahead of those.

## Vision

Bring diffusion-based text-to-image generation to MLPL, taught the MLPL
way: the *algorithm* (forward noising + reverse denoising) runnable in
the browser on tiny data, scaling up to a real Stable Diffusion
text-to-image pipeline on a connected GPU peer. Diffusion is the
non-autoregressive counterpart to next-token generation (see
[[Autoregression]] in the glossary): instead of emitting one token at a
time, it starts from noise and iteratively denoises a whole image.

## What MLPL already has vs. what SD needs

| SD component | MLPL today | Gap |
| --- | --- | --- |
| Text encoder (CLIP transformer) | `embed`, `attention`, `rms_norm`, `linear`, `chain` | tokenizer/vocab + cross-attention wiring |
| U-Net denoiser | -- | `conv2d` (+ transpose/upsample), `group_norm`, `silu`, cross-attention, timestep embedding |
| VAE decoder (latent -> pixels) | -- | conv stack (same `conv2d`) |
| Diffusion scheduler (DDPM/DDIM) | `while`/`repeat`, array ops, `sample` | a noise-schedule helper; the loop itself is pure MLPL |
| Pretrained weights (~GB safetensors) | safetensors is a transitive candle dep | an MLPL-level weight loader |
| GPU execution | `device("cuda")` / `device("mlx")` | conv / group-norm kernels on-device |

The load-bearing gap is `conv2d` (with autograd backward). MLPL is
conv-free today -- even the ViT demos use `patchify` + attention, not
convolution.

## Primitives to add (priority order)

1. `conv2d` (+ `conv_transpose2d` or `upsample_nearest`) -- forward +
   autograd backward. candle provides it on the GPU; the CPU backward
   must be written. THE load-bearing new primitive.
2. `group_norm` (SD's U-Net norm); `silu` = `x * sigmoid(x)` (trivial);
   `sin` / `cos` (for sinusoidal timestep embeddings -- add if missing).
3. `noise_schedule` helper (linear / cosine betas -> alphas, alpha-bar),
   or compose from `linspace` + `running_product`.
4. Cross-attention -- a `ModelSpec` variant (attention whose K/V come
   from text embeddings) or an `apply`-with-context form.
5. Pretrained-weight loader (safetensors -> `DenseArray` / device
   tensor) -- needed for real SD weights.

## Glossary terms to add

Diffusion model; Forward (noising) / Reverse (denoising) process;
DDPM / DDIM; Noise schedule (beta / alpha / alpha-bar); U-Net; Latent
diffusion; VAE; Cross-attention; Classifier-free guidance; Timestep
embedding; CLIP text encoder; Convolution / Conv2d; Group normalization;
SiLU. Cross-link to [[Autoregression]] (diffusion = the
non-autoregressive generation alternative).

## Demos and the browser-vs-connect split

Browser (live demo, user CPU/WASM) -- the diffusion *concept*, no conv,
no big weights:

- "Diffusion on 2D points" (two-moons / spiral): animate the forward
  noising over a schedule, train a tiny MLP denoiser live, run the
  reverse sampler. Teaches betas/alphas + the reverse loop with
  primitives MLPL already has (+ the noise-schedule helper). Fully
  in-browser.
- Noise-schedule plots; a "denoise one step" visualization.

Connect (MLX / CUDA) -- anything image-scale or real:

- "Image diffusion (MNIST-scale U-Net)": a small `conv2d` U-Net trained
  + sampled on the GPU (needs `conv2d` + `group_norm` on-device).
- "Stable Diffusion text-to-image" (headline): real CLIP + U-Net + VAE
  with pretrained weights. Conv-heavy + ~GB weights + a 20-50-step loop
  => GPU + server only. Pragmatic path: the CUDA backend already uses
  candle, so add `candle-transformers` (ships a Stable Diffusion impl)
  and expose it server-side, with MLPL orchestrating, e.g.
  `device("cuda") { text_to_image("a cat", steps=30) }`.

Why the split: browser WASM is CPU-only with no big-weight budget --
fine for the *algorithm* on tiny data, hopeless for 512px SD (GB
weights, conv-heavy, minutes/step on CPU). A GPU fast path + a server
for weights is mandatory for real text-to-image.

## Phasing (each ~a saga step group)

1. `conv2d` + autograd (CPU + CUDA/MLX) + glossary (Convolution, Group
   norm, SiLU) + the 2D-point diffusion browser demo (noise schedule,
   reverse loop). Smallest new surface that proves the concept.
2. Tiny U-Net image diffusion (connect): `group_norm`, `silu`, upsample,
   timestep embedding; MNIST-scale on the GPU.
3. Real Stable Diffusion (connect): `candle-transformers` SD on the
   CUDA/MLX server + pretrained-weight loader + cross-attention + CLIP;
   `text_to_image` demo + literate page (true-GPU, like the LoRA pages).

## What builds on this / relation to other plans

- Reuses the connect-server + device-gating from cuda-foundation and the
  `device("cuda")` / `device("mlx")` dispatch.
- The weight-loader + `candle-transformers` path also helps the parked
  `:ask` / local-LLM and the speculative-decode sagas (shared pretrained
  loading).
- Sequencing suggestion (supersedes the order in
  `future-sagas-multibackend-specdecode.md` for now): stable-diffusion
  Phase 1-2 next, then multi-backend-connect, then the rest.

## Risks / decisions

- CPU conv backward is fiddly; keep the in-browser demo MLP-only (no
  conv) so Phase 1 ships without a CPU conv-backward dependency, and
  gate the conv U-Net demos to connect mode.
- SD weights are large (GB) -- the loader needs streaming + an
  allow-listed fetch policy (mirror the SmolLM2 download policy in
  `docs/saga-local-gpu-agentic.md`).
- `candle-transformers` is a heavier dep; add it ONLY to the
  CUDA/MLX serve build (feature-gated), not the WASM bundle, to keep the
  browser build small and the disk footprint contained (see the
  disk-aware build notes in CLAUDE.md).
- Determinism for literate publishing: fix seeds so the published
  text-to-image output is reproducible.
