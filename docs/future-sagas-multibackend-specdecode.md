# Future sagas: multi-backend connect + speculative decoding

Planning input for 2-3 future sagas, beyond the demo matrix in
`docs/gpu-demos-roadmap.md`. Each section is structured to seed an
`agentrail init` (name, vision, steps, what it builds on, risks).

---

## Saga: `multi-backend-connect`

**Goal.** One web UI, served by one server, connected to *two or more*
back-end peers at once -- so a single dropdown enables demos for every
device any connected peer offers. Example: serve the UI on this Linux +
CUDA box, which itself runs CUDA demos in-process AND forwards MLX demos
to an Apple-Silicon peer; the dropdown lights up both the CUDA and MLX
groups, each routed to the right backend.

**What exists to build on.**

- `mlpl-serve` already parses *repeatable* `--peer <device>=<url>` into a
  `PeerRegistry`, and `RemoteMlxDispatcher` already forwards a
  `device(...)` block to the named peer over HTTP
  (`/v1/sessions/:id/eval-on-device`, `/transfer`). This is the
  orchestrator half -- it was built for the `mlx_remote` pattern.
- `GET /v1/devices` reports the *in-process* device set (cuda-foundation
  step 7). The web UI probes it and gates via
  `demo_disabled(cap, connected, peer_devices)`.

**Chosen architecture: server-as-orchestrator (reuse the registry).**
The web UI connects to ONE server; that server fans out `device(...)`
blocks to its `--peer` backends (and runs its own in-process device
directly). Cleaner than a client-side multi-connect because it reuses
the existing peer registry + remote dispatcher + the single
same-origin connect path (no per-peer CORS/TLS in the browser).

**Steps (draft).**

1. **aggregate-devices** -- `GET /v1/devices` returns the UNION of the
   server's in-process devices + each `--peer`'s probed `/v1/devices`
   (the server probes its peers at startup / on demand). So a Linux+CUDA
   server with `--peer mlx=http://mac:6465` reports
   `["cpu","cuda","mlx"]`. Tag each device with its source peer for
   routing + display.
2. **route-by-device** -- ensure `device("cuda")` runs in-process while
   `device("mlx")` forwards to the mlx peer (generalize
   `RemoteMlxDispatcher` from "mlx-only" to "any device -> its peer";
   in-process for the local feature, remote for the rest). Parity-test
   that a forwarded block round-trips.
2b. **peer health + fallback** -- if a peer is down, its devices drop
   out of `/v1/devices` (demos re-gate to disabled) rather than erroring
   mid-eval. Surface peer status in the UI.
3. **ui-multi-peer** -- the dropdown already gates by the probed set;
   verify both groups (CUDA + MLX) enable from one UI when the server
   has both a local GPU and a peer. Show which peer each group routes
   to. (The `use_peer_devices` hook already consumes `/v1/devices`; this
   is mostly server-side once aggregation lands.)
4. **lan-verify** -- end-to-end: Linux+CUDA server + `--peer
   mlx=<mac>`, browser on a third LAN machine, run a CUDA demo and an
   MLX demo from the same dropdown.

**Relation to the roadmap.** This subsumes the roadmap's
`dual-backend-connect` (saga 4) "enable both from one UI" half; the
*side-by-side comparison* half (run the same workload on two peers and
diff) is a natural follow-on once routing + aggregation land.

**Risks.** Cross-peer auth/token handling (each peer has its own
sessions); peer-probe latency on the aggregate endpoint (cache it);
device-name collisions if two peers offer the same device (the later
`nway-gpu-compare` saga needs per-peer keys like `cuda_a`/`cuda_b`).

---

## Saga: `speculative-decode-cpu` (goal 2a)

**Goal.** Speculative decoding with TWO models at once on a CPU box with
many cores + RAM: a small **draft** model (~0.5G) proposes K tokens; a
larger **target** model (~1G) verifies them in one batched forward;
accepted tokens advance, the first rejected token is resampled from the
target. Measure tokens/sec and acceptance rate vs. plain target-only
decoding -- the speedup is the point.

**What exists to build on.** MLPL already binds multiple models in one
session; `generate`/sampling + the tokenizer exist; `experiment` blocks
log metrics. The algorithm (draft K -> target verify -> accept/reject ->
resample) is expressible in MLPL control flow, or as a narrow builtin.

**Steps (draft).**

1. **specdecode-algorithm** -- implement the draft/verify/accept loop.
   Decide: pure-MLPL (a `while` loop over `generate` + a verify/compare)
   vs. a `speculative_decode(draft, target, prompt, k)` builtin. Prefer
   pure-MLPL first (showcases the language); fall back to a builtin if
   the per-step interpreter cost dominates. Assert correctness:
   spec-decode output == target-only output (same distribution /
   greedy-identical), only faster.
2. **specdecode-metrics** -- acceptance rate per step, tokens/sec,
   wall-clock vs target-only. A demo that prints both and the speedup.
3. **two-tiny-models** -- wire two real small models (draft 0.5G /
   target 1G) via the allow-listed `fetch_pretrained` path (see the
   SmolLM2 download policy in `docs/saga-local-gpu-agentic.md`).
   CPU-resident; lots of cores/RAM is the target environment.
4. **specdecode-demo** -- CLI + connect web demo + org page, measurable
   speedup on held-out prompts.

**Risks.** Real speedup needs the target's batched verify to be cheaper
than K serial target steps -- depends on the interpreter's batch matmul
efficiency (may need the `interp-alloc` perf work from
`docs/correctness-performance.md`). Tokenizer/vocab must match between
draft + target (or a mapping). Start with tiny synthetic models to prove
the algorithm before real weights.

---

## Saga: `speculative-decode-multigpu` (goal 2b)

**Goal.** The same speculative decoding, scaled to larger models on a
**two-GPU** box (e.g. 2x P100-16G): draft on one GPU, target on the
other (or both on one with the other for a second stream), for GPU
speculative-decoding experiments.

**What this needs that doesn't exist yet.**

- **Device indexing.** Today `device("cuda")` = CUDA device 0
  (`mlpl-cuda-rt::cuda_device` opens `new_cuda(0)`). Need
  `device("cuda:0")` / `device("cuda:1")` -> `candle Device::new_cuda(n)`,
  threaded through the dispatch + a per-index device cache. This is the
  load-bearing new capability.
- **Per-model placement.** `to_device(draft, "cuda:0")` /
  `to_device(target, "cuda:1")` so each model's params live on its GPU;
  cross-GPU tensor moves where the accept/reject step compares.
- **Concurrency.** Optionally run draft (GPU0) and the next target
  verify (GPU1) overlapped; candle ops are per-device, so two streams
  can progress in parallel.

**Steps (draft).**

1. **cuda-device-index** -- parse `device("cuda:N")`; `cuda_device(n)`
   cache (`OnceLock<HashMap<usize, Device>>`); dispatch + `to_device`
   carry the index. Test on this 1-GPU box that `cuda:0` works and
   `cuda:1` errors cleanly (no second GPU here).
2. **two-gpu-placement** -- place draft/target on separate indices;
   cross-GPU gather for verify. **Needs the 2-GPU box to test** -- this
   saga is structured here but its parity/perf tests run on the
   2x P100 host, not this single-GPU machine.
3. **specdecode-multigpu-demo** -- larger draft/target, measured
   tokens/sec + acceptance on the 2-GPU box.

**Sequencing note.** `speculative-decode-cpu` proves the *algorithm*
(testable here / on any CPU box); `speculative-decode-multigpu` adds the
*device-indexing capability* (step 1 testable here; steps 2-3 need the
2-GPU hardware). Could be merged into one `speculative-decode` saga if
preferred -- the split exists so the algorithm lands without waiting on
the 2-GPU host.

---

## Suggested order

1. `cuda-demo-parity` (in progress) -- finish the CUDA demo matrix.
2. `multi-backend-connect` -- one UI, many peers (CUDA + MLX together).
3. `speculative-decode-cpu` -- the algorithm + metrics on tiny models.
4. `speculative-decode-multigpu` -- device indexing + the 2-GPU runs.

Each is one `agentrail init` away; this doc is the seed input.
