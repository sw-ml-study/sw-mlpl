# demo-memory upstream requests

Tracking doc (planning). The demo-memory companion
(`companion-demo-memory.md`) states executable needs; this
records them and their status, following the mlplunit model.
Ranked as the agent reported them.

| # | Request | Status | Notes |
|---|---|---|---|
| 1 | High-resolution monotonic clock | SHIPPED (`clock_ms()`) | Native + connect only. Unblocks inserts/sec, lookups/sec, latency percentiles, wall-clock comparisons. |
| 2 | Fixed-width unsigned ints + bit ops | SHIPPED (band/bor/bxor/bnot/popcount/shl/shr/bmask/bits/from_bits) | mlpl-runtime-bits crate; design docs/bit-ops-design.md. Universal (browser+native+connect). |
| 3 | Packed layouts + observable storage size | QUEUED (`packed-layouts`) | Blocks credible bytes-per-key, alignment, cache-locality, tiny-pointer claims. Logical tiny-pointer navigation demonstrable before this lands. Design-first (touches the memory model). |
| 4 | First-class seeded RNG state | QUEUED (`rng-streams`) | Does NOT block (workloads are deterministic today via seeded randn/sample). Would make randomized hashing, shuffling, workload generation, and reproducible substreams COMPOSABLE. |
| 5 | Stable generation-state API + backend telemetry | PAUSED SAGA (`gen-state-kv-cache`) | Blocks only the end-to-end KV-cache acceleration demo, not classical hash/Bloom/LRU/retrieval. gen_state/gen_logits/gen_append already shipped (core); gen-controls + telemetry remain. |

## Sequencing

Clock first (done): it unblocks the self-measuring benchmark
story that distinguishes demo-memory. Fixed-width ints + bit ops
(SHIPPED) were the largest classical-structures unlock (Swiss,
Bloom, Hamming). Packed layouts are design-first and can wait
behind logical demonstrations. RNG streams are a composability
nice-to-have. The KV-cache demo waits on resuming the paused
gen-state saga.

Each request follows the discipline: the demo proves the need,
the core grows the smallest surface, the capability becomes
general rather than demo-specific.
