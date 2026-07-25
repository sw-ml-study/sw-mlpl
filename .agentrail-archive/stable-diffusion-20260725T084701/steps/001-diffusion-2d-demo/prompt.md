Step 1 of stable-diffusion: ship the diffusion ALGORITHM in the browser (tiny, no conv).

TDD throughout (Red/Green/Refactor).

1. Two small pure-array builtins the noise schedule needs (with unit tests
   first): linspace(start, stop, n) -> n evenly spaced values inclusive;
   cumprod(v) -> running product along a 1-D vector. Register in the runtime
   builtins + dispatch; help/describe entries.
2. Glossary terms in docs/glossary.md (alphabetical, ASCII, [[links]]):
   Diffusion model; Forward (noising) process / Reverse (denoising) process;
   DDPM; Noise schedule. Cross-link [[Autoregression]] (non-AR generation).
3. demos/diffusion_2d.mlpl + a "Diffusion (2D points)" web-demos registry
   entry (CPU/live tier): moons dataset -> linear beta schedule (linspace +
   cumprod for alpha-bar) -> forward-noise -> train a tiny MLP to predict the
   added noise (chain/linear/relu + adam + mse) -> reverse-sample from noise
   back onto the manifold -> scatter before/after.
4. TDD eval test: the trained denoiser's reverse samples land closer to the
   data manifold (smaller mean distance) than the untrained baseline.

Browser-runnable (no conv, no GPU, no big weights). Quality gate: cargo test,
clippy -D warnings, fmt, markdown-checker (glossary), sw-checklist, and rebuild
pages so the live demo + Glossary tab include it. Keep the in-browser demo
MLP-only so no CPU conv-backward is needed (conv lands in step 2).
