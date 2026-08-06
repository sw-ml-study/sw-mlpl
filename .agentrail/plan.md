# Saga: gen-state-kv-cache (resumed)
Track 2 opener (docs/future-sagas-queue.md item 4): the KV-cache
engine under the MTP generation-speed program. Full design:
docs/kv-cache-design.md (approved 2026-08-05; recommendations
stand per user direction 2026-08-06). Design step already done
in the archived run (.agentrail-archive/gen-state-kv-cache-
20260805T171947/); this resume starts at the core.
## Steps
1. gen-state-core -- GenerationState value + gen_state/gen_logits/
   gen_append (single token), CPU, TDD equivalence (bit-identical
   greedy vs recompute over the Tiny LM chain).
2. gen-controls -- gen_clone/gen_reset/gen_stats, multi-row
   append (verification hook), tutoring errors, :describe.
3. bench-and-demo -- benchmarks.md wall-clock table + the KV
   Cache demo (ids-equal proof + attended-positions cost curve).
4. mlx-resident-kv -- K/V as TensorHandles via dev_concat;
   fp32-tolerance equivalence; crossover bench.
5. close -- docs, queue advance to mtp-training, wiki errata.
