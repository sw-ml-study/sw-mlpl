# Future sagas queue

User-directed roadmap items queued 2026-08-01 (mid saga E4). Each
becomes its own agentrail saga with its own plan at start; order
within this list is roughly intended sequence, but items 3 and 4
are gated on moving back to the Linux/CUDA dev machine.

## 1. wiki-errata -- clean up the sw-mlpl.wiki errata page

The GitHub wiki lives in a SEPARATE repo checked out as a sibling
(`../sw-mlpl.wiki`; pages like `Apple-Silicon-and-MLX.md`,
`Capability-Matrix.md`). The errata page has accumulated stale and
wrong entries. Saga scope: audit every errata entry against the
current code (many were fixed by the E-series and decomposition
sagas), fix or delete stale ones, cross-link to CHANGES.md /
docs/saga.md where an erratum was resolved, and sweep the other
wiki pages for claims invalidated by Metal-on (the wiki's MLX page
predates saga E4: MLX now runs on the GPU, not Accelerate).
Remember: wiki commits push to the wiki repo, not this one; the
saga record still lives here.

## 2. mhc -- manifold-constrained Hyper-Connections (major feature)

Design source: `docs/mHc-research.txt` (feasibility analysis +
plan; see also the blog series at
software-wrighter-lab.github.io 2026-02-01 deepseek-papers-part1-mhc
and the reference implementations in `softwarewrighter/mHC-poc`,
which has equivalent MLX + PyTorch versions and a stability
benchmark). mHC generalizes residual connections to MULTIPLE
streams (`[batch, tokens, streams, features]`) whose per-layer
mixing matrices are projected onto a constrained manifold
(Sinkhorn-style doubly-stochastic projection built from exp,
reductions, clamp/maximum, division, transpose, matmul -- every
primitive already exists in the CPU runtime and the backends).

Per the research doc, treat it as TWO language capabilities, not
one built-in: (a) general constrained tensor transformations (the
projection as reusable ops), and (b) a high-level multi-stream
residual model layer in the Model DSL (a `ModelSpec` citizen like
Engram, with tape lowering and near-identity init). CPU-first with
demos (the Engram E1-E3 shape: primitives -> DSL -> demo), then
MLX riding the E4 TensorHandle residency work; CUDA parity is item
4 below. Expect the saga plan to pin CPU/MLX parity tolerances the
same way train_mlx_tests does.

## 3. engram-cuda -- finish Engram support on CUDA (Linux box)

docs/engram-sagas-plan.md E10, deferred until the dev host moves
back to the Linux/CUDA machine: Candle-CUDA first behind the same
frozen `mlpl-engram-core` addressing contract (golden fixture in
hash_tests), cudarc kernels only after profiling. Ride-along
verifications owed on that machine from the Mac era: the E3
cuda-target-gating manifests (candle behind the linux/x86_64
target table -- resolution semantics should be unchanged, run the
cuda workspace suites), and the /mw-cp playbook note about
--all-features being portable again once verified.

## 4. mhc-cuda -- finish mHC on CUDA (Linux box)

After items 2 and 3: bring the mHC layer up on the CUDA backend
(same backend-contract pattern as engram-cuda; the E4
TensorHandle/DeviceOps seam is backend-agnostic, so a
`mlpl-cuda-handle` implementing DeviceOps over candle tensors is
the natural shape). Includes CUDA-side parity with the mHC-poc
stability benchmark.

## Already-queued items this list does NOT repeat

- The crate-partition spike that retires the two standing
  sw-checklist crate-module-count FAILs (mlpl-eval,
  mlpl-eval-models) -- see docs/sw-checklist-paydown.md.
- The upstream agentrail fix for em dashes emitted by `agentrail
  instructions apply` into the managed CLAUDE.md block.
- Saga E4 remainder (backward residency, optimizer residency,
  bench gate) and E5-E9 per docs/engram-sagas-plan.md.
