# `POST /v1/eval-on-device` Contract (Saga R1)

## Status

- **Step 001:** stub returning 501.
- **Step 002 (shipped):** real implementation
  documented below. The wire format module +
  per-peer handle store + `Value::DeviceTensor`
  variant land together. See
  `contracts/eval-contract/device-tensor.md` for
  the orchestrator-side variant.
- **Step 003 (shipped):** orchestrator
  (`mlpl-serve`) `--peer mlx=<url>` flag +
  immutable peer registry.
- **Step 004 (shipped):** orchestrator
  `PeerDispatcher` hook forwards
  `device("mlx") { ... }` blocks to registered
  peers, stores returned tensor handles as
  `Value::DeviceTensor`, and materializes them
  with `to_device("cpu", x)`.

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

## Orchestrator flow (step 004, shipped)

1. `mlpl-serve --peer mlx=<url>` builds a peer
   registry keyed by device name. Non-loopback
   peer URLs require `--insecure-peers` in R1.
2. Before each `/v1/sessions/{id}/eval`, the
   handler installs a `PeerDispatcher` on that
   session's `Environment`; it is cleared after
   the eval call.
3. When eval sees `device("<name>") { body }`, it
   renders `body` back to MLPL source via
   `Display for Expr`, collects any array
   bindings whose names appear in that source,
   and calls the dispatcher before falling back
   to in-process device dispatch.
4. The dispatcher lazily creates a peer session by
   POSTing `/v1/sessions` on the peer, caches the
   peer session id + bearer token by peer URL, and
   then POSTs `/v1/sessions/{peer_id}/eval-on-
   device`.
5. A tensor response becomes
   `Value::DeviceTensor { peer, handle, shape,
   device }` in the orchestrator. Assignment binds
   that opaque value in the session environment.
6. CPU use of the opaque value strict-faults until
   the user calls `to_device("cpu", x)` (the older
   `to_device(x, "cpu")` form is still accepted).
   Fetch calls `/v1/sessions/{peer_id}/transfer`,
   decodes the tensor envelope, rebinds `x` as a
   local `Value::Array`, and stamps its placement
   as CPU.

The free-name binding collection is intentionally
substring-based for R1; AST-level scope analysis is
a follow-up once the protocol shape is stable.

## Endpoints (step 002, shipped)

### `POST /v1/sessions/{id}/eval-on-device`

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

Behavior:
1. Auth + session lookup (404 / 401 on miss).
2. Decode each binding via the wire format module
   (`decode_from_json` -> `DenseArray`); bind
   into a fresh sub-Environment.
3. Lex + parse + `eval_program_value` against the
   sub-Environment.
4. Inspect the result:
   - `Value::Array(a)` -> stash on the peer's
     handle store under a fresh uuid; return
     `{result: {kind: "tensor", handle, shape,
     device: "mlx"}}`.
   - `Value::Str(s)` -> return
     `{result: {kind: "string", value: s}}`.
   - `Value::Model(_)` / `Value::Tokenizer(_)` /
     `Value::DeviceTensor { .. }` -> 400 with the
     "must return a tensor or string" message.

### `POST /v1/sessions/{id}/transfer`

Authenticated. Pull the bytes back to the
orchestrator. Request:

```json
{"handle": "<uuid>"}
```

Response on success:

```json
{"tensor": "<base64-bincode>"}
```

The orchestrator decodes via `decode_from_json`
and rebinds as a local `Value::Array`.

### `POST /v1/sessions/{id}/release-handle/{handle}`

Authenticated. Idempotent-style cleanup of a
single handle. Returns `204 No Content` on
release; `404 Not Found` if the handle is unknown
(already released or never existed). Session-drop
also implicitly releases all handles owned by
that session (step 003 will wire the actual
session-drop hook from the orchestrator side).

### Error responses (all endpoints)

- `400 Bad Request` -- malformed body, lex/parse
  error, eval error, malformed binding (bad
  base64 / bincode / shape mismatch), or block
  evaluated to a non-tensor / non-string Value.
- `401 Unauthorized` -- missing or wrong bearer.
- `404 Not Found` -- unknown session id, or
  unknown handle on transfer / release-handle.

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

`to_device("cpu", x)` is the canonical way to
materialize the bytes back (`to_device(x, "cpu")`
remains accepted for the pre-R1 call order) -- it
calls the peer's
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
