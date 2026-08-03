# Saga E5: engram-mlx (Engram on the resident TensorHandle seam)

Per docs/engram-sagas-plan.md E5 row and the CONSTRAINED scope in
docs/future-sagas-queue.md Track 0: resident addressing + gather +
projection/gating, CPU/MLX parity + perf (seam counters), the
Tiny-LM Engram demo on MLX. Do NOT touch E6-E9 scope (sparse
IndexedRows optimizer, 100M tables, imports).

State (verified 2026-08-02): engram_tape (mlpl-models-tape
layers.rs) lowers apply_engram to selection-matmul + reshape +
linear + concat + sigmoid ops. Under the E4 resident tape:
hashing stays host (frozen cross-backend bit contract -- cheap,
deterministic); the selection one-hot is a host leaf uploaded per
step; sel@memory matmul, reshape, linears, sigmoid/mul/add run
resident; backward scatter-ADD is sel^T @ upstream = resident
matmul. The known seam breaks: concat (structural CPU fallback,
forces h+v down and re-uploads, both forward and backward) and
the per-step selection-matrix upload. mlx-rs surface available:
concatenate, split (concat backward), take/take_axis (gather).

Perf reality from E4: per-op dispatch floor ~35us/submission;
CPU wins below ~d=128. The engram gate is therefore honest
profile + parity + a documented crossover, not a blanket
"MLX faster" claim at toy sizes.

## Steps

1. baseline-and-parity -- seam-profile the existing
   engram-in-chain resident training (engram_stats parity,
   hash-contract bit-exactness CPU vs MLX, N-step trajectory
   parity test at fp32 tolerance); record per-step
   uploads/downloads/submits/fallbacks; identify the concat
   fallback + selection upload in the numbers. TDD: the parity
   tests are the red/green; profile documented in the step
   summary (benchmarks.md gets the final table in step 4).
2. dev-concat -- DeviceOps grows concat (axis) + its backward
   split: MlxOps via mlx concatenate/split_axis; TensorHandle
   dev_concat/dev_split; resident forward attempt for Concat
   nodes + resident backward; parity + no-download pinning tests
   (resident_seam_tests pattern). Also benefits the attention
   path (multi-head concat).
3. resident-gather-eval -- evaluate replacing the selection
   one-hot (host-built [T*orders*heads, rows] leaf uploaded per
   step) with a device gather: DeviceOps::gather_rows via mlx
   take_axis IF a resident scatter-add backward is available in
   vendored mlx-rs (check indexing/indexmut surface); otherwise
   KEEP the selection-matmul (it is already resident with
   resident scatter-add backward) and instead cache the selection
   leaf across steps when ids are unchanged (the demo trains on a
   fixed context window -- the upload is redundant). Measure both
   candidates with the seam counters; ship the winner; document
   the decision honestly.
4. perf-gate-and-demo -- mlx_vs_cpu bench gains an
   engram-in-chain train-step workload (demo scale + a
   crossover-scale variant) with seam printout; the Tiny-LM +
   Engram MLX demo (native demos/ script + web demo entry with
   the same device gating as the other MLX demos); benchmarks.md
   E5 table; docs + wiki per discipline. Acceptance: parity
   green, zero unexplained fallbacks in the engram step (CE
   backward only), crossover documented, demo runs on MLX.
5. close -- saga.md entry, engram-sagas-plan E5 row status,
   queue update, wiki errata; --done.
