# Saga: Stable Diffusion (text-to-image)

Active-saga plan. Full scoping + rationale lives in
`docs/future-saga-stable-diffusion.md`; this is the executable step
breakdown. Diffusion is the non-autoregressive generation path (see the
glossary [[Autoregression]] entry): start from noise, iteratively
denoise.

Strategy: ship the diffusion *algorithm* in the browser first (tiny,
no conv), then add `conv2d` for image U-Nets (GPU), then real Stable
Diffusion via candle-transformers on a connected GPU peer.

## Steps

1. **diffusion-2d-demo** -- the diffusion algorithm, browser-runnable,
   no conv. Add the two small pure-array builtins the noise schedule
   needs: `linspace(start, stop, n)` and `cumprod(v)` (TDD: unit tests +
   gradcheck where applicable). Glossary: Diffusion model, Forward
   (noising) / Reverse (denoising) process, DDPM, Noise schedule. A
   `demos/diffusion_2d.mlpl` + web-demos registry entry: forward-noise a
   2D dataset (moons) over a linear beta schedule, train a tiny MLP to
   predict the added noise, reverse-sample from noise back onto the data
   manifold, scatter before/after. TDD: an eval test that the trained
   denoiser's reverse samples land closer to the manifold than the
   untrained baseline.

2. **conv2d-primitive** -- `conv2d` (+ `conv_transpose2d` or
   `upsample_nearest`) forward + autograd backward on CPU and the
   device backends; `group_norm`, `silu`, `sin`/`cos` if missing.
   Glossary: Convolution / Conv2d, Group normalization, SiLU. Gradcheck
   parity (CPU) + device parity.

3. **tiny-unet-diffusion** -- a small conv U-Net image diffusion
   (MNIST-scale), trained + sampled under `device("cuda")` /
   `device("mlx")`. Timestep embedding, group_norm, silu, upsample.
   Connect-only demo + literate page.

4. **stable-diffusion-connect** -- real text-to-image: add
   `candle-transformers` to the CUDA/MLX serve build (feature-gated),
   a safetensors pretrained-weight loader, cross-attention + CLIP
   wiring, and a `text_to_image(prompt, steps)` form under
   `device("cuda")`. Connect-only demo + true-GPU literate page. May
   split into sub-steps (weights loader; CLIP+U-Net+VAE wiring; demo).

## Browser vs connect

- Step 1 runs fully in the browser (CPU/WASM): tiny MLP denoiser, no
  big weights, no conv.
- Steps 3-4 are connect-only (conv-heavy / GB weights -> GPU + server),
  gated by the peer's real device like the other CUDA/MLX demos.

## De-risking

- Keep step 1's in-browser demo MLP-only (no conv) so it ships without a
  CPU conv-backward dependency; conv lands in step 2 and the conv demos
  are GPU/connect-gated.
- `candle-transformers` is added ONLY to the serve build (feature-gated),
  never the WASM bundle (disk + bundle size).
- Fixed seeds so literate publishing is reproducible.

## References

- `docs/future-saga-stable-diffusion.md` (scoping + risks).
- The MLX/CUDA LoRA demos + literate pages (the connect + true-GPU
  literate pattern to mirror for steps 3-4).
