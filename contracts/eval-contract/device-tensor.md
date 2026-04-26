# `Value::DeviceTensor` Contract (Saga R1 step 002)

## Purpose

A peer-resident tensor reference. The bytes live on
the peer named by `peer`; the orchestrator only
holds the handle plus shape plus device metadata.
Returned by `mlpl-mlx-serve`'s `eval-on-device`
endpoint when a block evaluates to a tensor; later
sagas (R2 CUDA, R3 distributed) reuse the variant
unchanged.

## Shape

```rust
Value::DeviceTensor {
    peer: String,        // peer URL the bytes live on
    handle: String,      // opaque uuid issued by the peer
    shape: Vec<usize>,   // cached shape so callers can
                         //   shape(x) without a network
                         //   round-trip
    device: String,      // "mlx" / future "cuda" / ...
}
```

## Strict-fault on cross-device ops

`Value::into_array` and `Value::as_array` on a
`DeviceTensor` return
`EvalError::DeviceTensorFault { peer, device }`,
not the generic `ExpectedArray`. The Display impl
on the error prints:

```
tensor lives on <peer>:<device>; \
  use to_device('cpu', x) to fetch
```

Any CPU op that tries to consume the tensor faults
with that message; the user must explicitly
`to_device('cpu', x)` to materialize the bytes
back. No auto-fetch in R1 -- the round-trip cost
should be visible in source.

## Display

```
<tensor on <peer>:<device>; shape=[<dims>]>
```

Used by REPL prints and error context. Includes
shape so users can sanity-check without
materializing the bytes.

## Materialization (`to_device('cpu', x)`)

Step 003 wires `to_device('cpu', x)` to call the
peer's `POST /v1/sessions/{id}/transfer` endpoint
(see
`contracts/serve-contract/eval-on-device.md`),
decode the returned base64-bincode envelope via
the wire format module, and rebind the value as a
`Value::Array` in the orchestrator's
`Environment`. Step 002 lands the variant + the
peer-side handler; step 003 wires the orchestrator
side.

## Lifetime

- Created peer-side when an `eval-on-device` block
  returns a tensor; the peer stashes the bytes in
  its handle store under a fresh uuid.
- The orchestrator binds the resulting
  `Value::DeviceTensor` in its session
  Environment.
- Released peer-side either via explicit
  `POST /v1/sessions/{id}/release-handle/{handle}`
  or when the orchestrator's session is dropped
  (which step 003's session-drop hook will trigger
  for every handle it owns on each peer).

## Non-goals (R1)

- **Auto-fetch on cross-device ops.** Strict-fault
  is the safe default; future sagas may add an
  opt-in auto-fetch with a warning.
- **Direct peer-to-peer migration.** R3
  optimization; today every cross-peer op routes
  through the orchestrator's CPU heap.
- **Persistence across orchestrator restarts.**
  Sessions die with the orchestrator; the peer's
  handle store can be GC'd at that point too.
- **f32 / bf16 / f16 dtypes.** R1 wire format is
  f64-only; the version field reserves space.
