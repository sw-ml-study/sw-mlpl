# Saga: gen-state-kv-cache (resumed at gen-controls)
Track 2 KV-cache program (docs/kv-cache-design.md). gen-state-
core shipped: Value::GenState, gen_state/gen_logits/gen_append
(single token), bit-identical greedy equivalence. Resuming at
gen-controls; new controls go in a SIBLING module
(fncall_gen_controls.rs) since fncall_gen.rs is at the module-
function ceiling (docs/sw-checklist-paydown.md).
## Steps
1. gen-controls -- gen_clone(gs), gen_reset(gs), gen_stats(gs)
   ({tokens, layers, kv_rows, kv_values}), multi-row
   gen_append (rank-1 id vector = batched verification hook);
   catalog + docs; TDD.
2. bench-and-demo -- benchmarks.md wall-clock (cached vs
   recompute, now that clock_ms exists) + a KV Cache web demo
   (ids-equal proof + attended-positions cost curve).
3. close -- docs, queue advance to mtp-training, wiki.
