# CNN Builtins + Visualization Milestone

Saga 39, proposed.

## Why this exists

MLPL has dense layers (linear, chain, residual) and
attention but no convolution. CNNs are the entry point for
image understanding and a prerequisite for the ViT demos
to show "what came before transformers." Adding conv2d and
pool2d completes the model-building surface for image tasks.

The 3D visualization stage can then render:
- Conv filters as small colored grids
- Feature maps as heatmap layers
- Pooling as downsampled versions
- The full CNN pipeline as a spatial data flow

## Builtins to add

| Builtin | Args | Description |
|---------|------|-------------|
| `conv2d(input, filters, stride, padding)` | 4 | 2D convolution. Input `[B, C_in, H, W]`, filters `[C_out, C_in, kH, kW]`. Returns `[B, C_out, H', W']`. |
| `pool2d(input, size, mode)` | 3 | Pooling. `mode` is `"max"` or `"avg"`. Size is `[pH, pW]`. Returns downsampled output. |
| `relu(x)` | 1 | Already exists as `relu_layer` in model DSL; add a standalone elementwise version. |

## 3D visualization

When conv2d runs with `:3d` on:
- Input tensor: stack of colored heatmap layers (one per channel)
- Filters: small colored grids (one per output channel)
- Output feature maps: stack of heatmap layers
- Connection arrows from input through filters to output

When pool2d runs:
- Input feature maps shrink to output size with a visual
  "compression" in the sculpture dimensions

## Steps

### Step 001 -- conv2d builtin

Implement `conv2d(input, filters, stride, padding)` in a
new `crates/mlpl-runtime/src/conv_builtins.rs`. Pure Rust
nested loops (no BLAS/SIMD -- correctness first, speed
later). Unit tests with known outputs.

### Step 002 -- pool2d builtin

Implement `pool2d(input, size, mode)` in the same file.
Max and average pooling. Unit tests.

### Step 003 -- relu standalone

Add `relu(x)` as an elementwise builtin (max(0, x)) in
math_builtins.rs. The model DSL `relu_layer()` stays for
composing layers; `relu(x)` is the standalone version.

### Step 004 -- CNN demo

A new demo "Simple CNN" that builds a 2-layer CNN:
conv2d -> relu -> pool2d -> conv2d -> relu -> pool2d ->
reshape -> linear -> softmax. Runs on a tiny synthetic
dataset (e.g., 8x8 images with simple patterns).

### Step 005 -- 3D conv visualization

Update stage3d.js to recognize conv2d/pool2d outputs and
render them as stacked heatmap layers. Each channel is a
separate layer in the stack, colored by element values.

### Step 006 -- Polish + close

Docs, help text, glossary entries, tour stop, pages
rebuild, saga close.

## Quality requirements

Same as saga 38. TDD for builtins. Warning-target design
for new modules.
