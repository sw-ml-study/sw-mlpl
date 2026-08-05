# Saga: gen-state-kv-cache
Track 2 opener (docs/future-sagas-queue.md item 4): the KV-cache
engine under the MTP generation-speed program. Full design:
docs/kv-cache-design.md. Exit: cached generation measurably
faster, greedy outputs bit-identical on CPU.
## Steps
1. design -- docs/kv-cache-design.md; pause for user review of
   the gen_* surface + three open questions.
2. gen-state-core -- GenerationState value + gen_state/gen_logits/
   gen_append (single token), CPU, TDD equivalence (bit-identical
   greedy vs recompute over the Tiny LM chain).
3. gen-controls -- gen_clone/gen_reset/gen_stats, multi-row
   append (verification hook), tutoring errors, :describe.
4. bench-and-demo -- benchmarks.md wall-clock table + the KV
   Cache demo (ids-equal proof + attended-positions cost curve).
5. mlx-resident-kv -- K/V as TensorHandles via dev_concat;
   fp32-tolerance equivalence; crossover bench.
6. close -- docs, queue advance to mtp-training, wiki errata.
