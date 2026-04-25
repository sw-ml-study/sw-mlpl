# `POST /v1/eval-on-device` Contract (Saga R1)

## Status

- **Step 001 (this saga):** STUB. The endpoint exists
  on `mlpl-mlx-serve` and returns `501 Not
  Implemented` with body `{"error": "wire format
  ships in Saga R1 step 002; ..."}`. Forward-points
  the implementation to step 002.
- **Step 002:** real implementation with the bincode
  + versioned-envelope tensor wire format described
  below.
- **Step 003:** orchestrator (`mlpl-serve`) `--peer
  mlx=<url>` flag wires `device("mlx") { ... }`
  blocks to call this endpoint.

## Purpose

`POST /v1/sessions/{id}/eval-on-device` runs an MLPL
program block against a device-bound `Environment`
on a peer server, with explicit input tensor
bindings shipped over the wire and a single result
tensor (or string) returned. The caller is the
orchestrator's `device("<name>") { ... }` block
forwarder; the callee is the device-specific peer
binary (`mlpl-mlx-serve`, future
`mlpl-cuda-serve`).

CLI-only LAN deployment in R1. The browser path
through this endpoint waits on R3's auto-discovery
+ per-peer auth work.

## Signature (step 002 target)

Authenticated. Request:

```http
POST /v1/sessions/<id>/eval-on-device
Authorization: Bearer <token>
Content-Type: application/json

{
  "program": "y = x * 2",
  "bindings": [
    {"name": "x", "tensor": "<base64-bincode>"}
  ]
}
```

Response on success:

```json
{
  "result": {
    "kind": "tensor",
    "handle": "<uuid>",
    "shape": [3, 4],
    "device": "mlx"
  }
}
```

Or, when the block evaluates to a string:

```json
{
  "result": {
    "kind": "string",
    "value": "<the string>"
  }
}
```

Error responses:
- `400 Bad Request` -- malformed body, lex/parse
  error, eval error, or block evaluated to a
  non-tensor / non-string Value (Model / Tokenizer
  not supported in R1).
- `401 Unauthorized` -- missing or wrong bearer.
- `404 Not Found` -- unknown session id.
- `501 Not Implemented` -- step 001 stub only.

## Tensor wire format (step 002)

`bincode` 1.x with explicit versioned envelope:

```
{
  version: u32 = 1,        // future dtype expansion
                           //   bumps this
  dtype: u8 = 0,           // 0 = f64; 1 = f32 (R1
                           //   does not implement
                           //   1, but the slot is
                           //   reserved)
  ndim: u8,                // rank
  shape: [u64; ndim],      // dims
  data: [u8]               // shape.iter().product()
                           //   * dtype_bytes raw
                           //   little-endian bytes
}
```

Endianness fixed little-endian (matches every
target architecture MLPL supports today). Future
saga additions:
- `dtype = 1` (f32), `dtype = 2` (bf16), `dtype = 3`
  (f16) when the dtype machinery ships (separate
  saga; out of scope for R1).
- Sparse tensor encoding (much later).

The tensor envelope is bincode-encoded, then
base64-encoded for the JSON-friendly transport
field. Future versions could ship a binary
`Content-Type: application/octet-stream` request
shape for large tensors; the JSON path is the
v1 minimum-viable transport.

## Strict-fault on cross-device ops

A `Value::DeviceTensor { peer, handle, shape,
device }` returned by this endpoint is opaque to
the orchestrator. Touching it from a different
device scope (e.g., a CPU op on the orchestrator
that consumes a tensor still resident on the MLX
peer) errors:

```
tensor lives on http://mac.local:6465:mlx; \
  use to_device('cpu', x) to fetch
```

`to_device('cpu', x)` is the only way to
materialize the bytes back -- it calls the peer's
`POST /v1/sessions/{id}/transfer` (also lands in
step 002) which returns the raw envelope so the
orchestrator can decode + rebind locally as a
`Value::Array`.

## Handle lifecycle

- A handle is created when the eval-on-device
  endpoint returns a tensor result.
- The peer holds the bytes in an in-memory
  handle store keyed by uuid.
- The orchestrator's session keeps the handle
  reference alive in its `Environment` as a
  `Value::DeviceTensor`.
- Cleanup: handles are released when the
  orchestrator's session drops, or via explicit
  `POST /v1/sessions/{id}/release-handle/{h}`
  before then.

## Non-goals (deferred)

These items are out of scope for Saga R1 and land
in R2 / R3 / a separate dtype saga / a Saga 21
follow-up:

- `bf16` / `f16` tensor dtype support (R1 wire
  format is f64-only; the version field reserves
  space for future dtypes).
- Sparse tensor encoding.
- Streaming + cancellation of long-running blocks
  (block-granularity RPC is sufficient for R1).
- Server-side LLM proxy (Saga 21 follow-up; LLM
  routing is unrelated to device routing).
- Peer-to-peer tensor migration without going
  through the orchestrator's CPU heap (R3
  optimization).
- mDNS / LAN auto-discovery of peers (R3).
- Per-peer bearer-token rotation (R3 auth design).
- CUDA device support -- ships in R2 via
  `mlpl-cuda-serve` reusing this contract.
