# Vendored mlx-rs 0.25.3

This directory holds an as-published copy of the `mlx-rs` v0.25.3
crate (MIT OR Apache-2.0, same as this repo) with **one line**
changed from upstream.

## Why vendored

`mlx-rs` 0.25.3 declares its `mlx-sys` dependency as:

```toml
[dependencies.mlx-sys]
version = "=0.2.0"
```

without `default-features = false`. Because `mlx-sys`'s default
feature set is `["accelerate", "metal"]`, Cargo's feature-union
rule keeps the `metal` feature on for `mlx-sys` no matter what
downstream crates (including `mlpl-mlx`) request. The `metal`
feature forces the MLX C++ CMake build to compile Metal shaders,
which in turn requires the `metal` / `metallib` shader compilers
that ship **only** with the full `Xcode.app` -- not the Command
Line Tools most Rust developers have.

For the MLPL Saga 14 step 001 scope (plumbing + parity test on
`matmul`) the Accelerate CPU/SIMD backend is enough: we're proving
that MLPL can swap `mlpl-rt` for `mlpl-mlx` at the primitive
boundary, not measuring GPU speedup (that's step 008).

## What changed

Exactly one edit to `Cargo.toml`:

```diff
 [dependencies.mlx-sys]
 version = "=0.2.0"
+default-features = false
```

Source code, build scripts, and Cargo.lock are identical to the
published crate. `Cargo.toml.orig` is the upstream pre-normalization
`Cargo.toml` and is kept verbatim for reference.

## How the workspace consumes it

Top-level `Cargo.toml` has:

```toml
[patch.crates-io]
mlx-rs = { path = "vendor/mlx-rs" }
```

which replaces the crates.io copy of `mlx-rs` with this vendored
one during dependency resolution. Everything that depends on
`mlx-rs` (currently just `crates/mlpl-mlx`) picks up the fixed
`mlx-sys` dependency transparently.

## Upstream fix

This is an upstream packaging bug. When `mlx-rs` publishes a
release with the one-line fix (or the `mlx-sys` author removes
`metal` from `mlx-sys`'s default features), this directory can be
deleted and the `[patch.crates-io]` entry dropped.

Tracking: file an issue at <https://github.com/oxideai/mlx-rs>
if one is not already open.
