# `calibrate_device` Contract (Saga 22 step 002)

## Purpose

`calibrate_device([size]) -> gflops` benchmarks matmul
throughput on the active device and caches the result
into `env.set_string("mlpl_device_throughput_gflops",
...)`. Subsequent `estimate_train` /
`estimate_hypothetical` calls read that key to produce
honest wall-clock numbers instead of the default
conservative 50 GFLOPS lower bound.

## Signature

```
calibrate_device()
calibrate_device(size)
```

- `size` -- optional positive integer scalar. Default
  `1024`. Runs `BENCH_ITERS = 10` square-matmul ops
  of shape `[size, size] @ [size, size]` after a
  single warmup pass.
- Returns a scalar f64: observed GFLOPS.
- Side effect: writes
  `env.mlpl_device_throughput_gflops` to the same
  value (as a string) so any subsequent estimator
  reads it.

## Device-awareness

The benchmark is dispatched through the same
`device::dispatched_call(env, "matmul", ...)` path as
every other matmul in MLPL, so
`device("mlx") { calibrate_device() }` measures MLX
throughput and overwrites the cached key with that
value. The key has no device suffix; switching devices
invalidates the cache and the user is expected to
re-calibrate. Keeping a single key (instead of a
device-suffixed pair) keeps `estimate_train`'s lookup
path trivial.

## What gets measured

Flat square matmul of shape `[size, size] @ [size,
size]`. Deterministic input data built from a small
integer modulus (`(i % 97) * 0.01` and `(i % 89) *
0.01`) so the benchmark does not depend on a PRNG or
on startup randomness.

Observed GFLOPS = `2 * size^3 * BENCH_ITERS /
elapsed_seconds / 1e9`. The first iteration is a
warmup and is not counted. Clock source:
`std::time::Instant`.

## Accuracy caveats

- Throughput varies 2x+ with matmul dimensions and
  data reuse; a 1024x1024 benchmark on f64 does not
  predict the throughput of (say) 32x32 Attention
  projections in a real training step. Conservative
  usage: calibrate at roughly the d_model you plan
  to train at.
- MLX throughput can be underreported on the first
  benchmark run of a session because of kernel
  cache misses.
- CPU throughput varies with load -- a busy
  background process halves the measured number.
- Default 1024 is chosen as a reasonable balance:
  big enough that overhead does not dominate, small
  enough to finish in under a minute on a laptop
  CPU.

## Error cases

- **Wrong arity** (> 1 arg).
- **Non-scalar size** / **non-positive size**:
  `"calibrate_device: size must be positive, got
  <value>"`.

## What this contract does NOT cover

- **Per-dtype calibration.** MLPL is f64; the cached
  number corresponds to f64 throughput. A future
  saga with f16/bf16 tensors will need a dtype-
  aware cache.
- **Memory-bandwidth vs compute-bound measurement.**
  `calibrate_device` is a pure-compute benchmark;
  it does not tell you whether your model will be
  memory-bandwidth-bound.
- **Per-operation calibration.** One matmul number
  does not predict per-softmax or per-layer-norm
  throughput. Estimators assume matmul dominates;
  this is typically true for Transformer training
  but can mislead for other workloads.
- **Multi-GPU calibration.** Single device only.
- **Streaming / warmup schedules.** `BENCH_ITERS =
  10` after one warmup iter; no more sophisticated
  statistics. A future enhancement could track
  per-iter variance and return a confidence
  interval.
