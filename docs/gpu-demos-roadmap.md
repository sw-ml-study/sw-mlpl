# GPU demos + multi-backend connect -- program roadmap

The grand arc: a complete, symmetric set of GPU demos across **MLX
(Apple)** and **CUDA (NVIDIA/Linux)**, each demo present as a CLI
script, a connect-mode web demo, and an org-mode literate doc; plus
a remote/transport demo per backend; building toward a UI that can
connect to **two backends at once** (MLX + CUDA) for side-by-side
comparison, and eventually to **two CUDA peers with different GPUs**
for N-way hardware comparison.

This doc is the program-level plan. Individual sagas
(`docs/saga-*.md`) implement slices of it.

## Demo taxonomy

### Group A -- ML demos (the matrix)

Four *distinct ML stories*, each with measurable learning on
held-out data (the project's demo success metric):

1. **LoRA fine-tune** -- full-GPU LoRA adapter fine-tune of a tiny LM.
2. **Tiny LM** -- pretrain a tiny language model (perplexity drop).
3. **Neural Thicket** -- ensemble variant sweep + inference.
4. **tic-tac-toe fine-tune** -- self-play RL fine-tune.

Target: each demo as **CLI + Web + org**, for **MLX + CUDA** =
`4 demos x 3 surfaces x 2 backends = 24 artifacts`.

### Group B -- remote / transport demos

Not ML demos -- they exercise the orchestrator/peer transport:
remote `device(...)` dispatch, opaque device-tensor handles, and
explicit `to_device` data movement across the process/network
boundary.

- **mlx_remote** (exists, CLI only) and **cuda_remote** (new twin).

Target: each as CLI + Web + org, grouped under a "Remote / Connect"
section in the UI (distinct from the ML demo groups) =
`2 demos x 3 surfaces = 6 artifacts`.

### Group C -- comparison demos (emergent, later sagas)

Side-by-side runs of one workload across multiple connected peers
(MLX vs CUDA; CUDA-A vs CUDA-B). These are views/demos that build
on the multi-backend connect work below.

## Current state (2026-06-02)

| Demo | CLI | Web | org | MLX status |
|------|-----|-----|-----|------------|
| LoRA fine-tune | yes | yes | yes | COMPLETE |
| tic-tac-toe | no  | yes | yes | needs CLI |
| Tiny LM | yes | no | no | needs Web + org |
| Neural Thicket | yes | no | no | needs Web + org |
| mlx_remote | yes | no | no | needs Web + org |

CUDA side: nothing yet (cuda-foundation builds the engine + the
first CUDA demo). Only **LoRA fine-tune** is currently complete on
any backend.

## Connect architecture arc

1. **Single connect** -- UI connects to ONE peer (Mac MLX or Arch
   CUDA); a `GET /api/devices` probe reports what that server
   offers (`cpu`/`mlx`/`cuda`) and the UI gates demos by it. (This
   is `cuda-foundation` step 6.)
2. **Dual backend** -- UI connects to an MLX peer AND a CUDA peer
   at once; run the same demo on both; render side-by-side
   (loss curves, wall-clock, output diff within tol). The
   orchestrator already holds multiple named peers
   (`--peer mlx=... --peer cuda=...`, `PeerRegistry`); the work is
   comparison orchestration + UI + aggregated `/api/devices`.
3. **N-way same-device** -- two+ CUDA peers with DIFFERENT GPUs
   (e.g. RTX 5060 Ti vs another card). Generalize the registry to
   multiple peers per device-type (keys like `cuda_a`, `cuda_b`)
   and add GPU identity (name, memory, compute-cap) to
   `/api/devices` so the UI labels each comparison column.

## Serving + LAN access (requirement)

The dev workflow needs a remote LAN client (e.g. a laptop) to reach
the connect server + web UI running on this Arch box. Reality today:

- `mlpl-serve --bind 0.0.0.0:6464` already binds on all interfaces;
  non-loopback binds REQUIRE `--auth required` (auth-disabled is
  loopback-only, a deliberate RCE guard -- the server evaluates
  arbitrary pasted MLPL).
- The same `mlpl-serve` serves the web playground via
  `--static-dir <web-dist>` at `<scheme>://<bind>/sw-mlpl/`.
- So one process is both the static UI host and the connect target.

The CUDA connect-peer step (saga 1, step 7) must therefore verify
the LAN path end to end, not just loopback: bind `0.0.0.0`, serve
the UI, connect a remote browser via `?connect=`, and confirm the
SSE eval stream + `/api/devices` work cross-host. The known gotcha
is TLS/CORS over the LAN: the self-signed cert is loopback-only, so
a LAN client needs either plain HTTP (browser mixed-content / SSE
caveats) or a cert valid for the box's LAN name/IP. Resolve this in
step 7.

## Cross-cutting: define-once consolidation

Today a demo's source can live in three places (the `.mlpl` CLI
file, an inline string in the web registry, and the `.org` doc),
which is why coverage drifted. The consolidation principle: **one
canonical `.mlpl` per demo is the source of truth**; the
web-registry entry and the `.org` literate doc are derived/extracted
from it by tooling. Introduced in the MLX consolidation saga so the
CUDA saga inherits it.

Also relevant: **3D viz passthrough** (see
`docs/saga-local-gpu-agentic.md` Phase 1c) -- comparison views need
viz data to flow back from peers, not just a display string.

## Saga roadmap

1. **cuda-foundation** (ACTIVE) -- CUDA engine (`mlpl-cuda-rt` ->
   dispatch -> forward/model -> train) + the first CUDA demo (LoRA:
   CLI + Web + org) + single-connect peer (`mlpl-cuda-serve` +
   `/api/devices`). Proves the whole vertical slice on this box.
2. **mlx-demo-consolidation** -- the define-once tooling, then fill
   the MLX gaps: tic-tac-toe CLI; Tiny LM Web + org; Neural Thicket
   Web + org; mlx_remote Web + org. Result: MLX = 4 ML x (CLI+Web+
   org) + mlx_remote x (CLI+Web+org). MLX runtime parity tests are
   Apple-gated (authoring is doable on Linux; execution needs a Mac).
3. **cuda-demo-parity** -- CUDA equivalents for the remaining 3 ML
   demos (tic-tac-toe, Tiny LM, Neural Thicket) + cuda_remote, using
   the define-once tooling. Result: CUDA = 4 ML x (CLI+Web+org) +
   cuda_remote. Fully testable on this box.
4. **dual-backend-connect** -- run one workload across MLX + CUDA
   peers simultaneously; side-by-side comparison view; aggregated
   `/api/devices`; a "Compare backends" demo (Group C).
5. **nway-gpu-compare** -- two+ CUDA peers with different GPUs;
   multi-peer-per-device registry + GPU-identity metadata; N-column
   comparison UI; a "two CUDA GPUs" benchmark demo.

After saga 3 the full `24 + 6` artifact matrix exists; sagas 4-5
add the comparison superpowers on top.

## End-state scorecard

- MLX: 4 ML demos x (CLI + Web + org) + mlx_remote x (CLI+Web+org)
- CUDA: 4 ML demos x (CLI + Web + org) + cuda_remote x (CLI+Web+org)
- UI: connect to 1 peer (gate by device); 2 peers (MLX vs CUDA
  compare); N CUDA peers (different GPUs compare).
