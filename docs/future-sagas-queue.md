# Future sagas queue

REPRIORITIZED 2026-08-02 per user direction, refining the
2026-08-01 ordering from `docs/project-direction.txt` (the
four-priority strategy: generation speed, agent quality, Colab
replacement, small/old hardware). The one change from the prior
version of this file: the mHC track no longer sits between the
substrate and the MTP program -- mHC-CPU is PARKED BEHIND MTP
(parked, not cancelled), and every mHC/Engram accelerator phase
beyond MLX-for-Engram is deferred. The strategic spine is now:
finish the execution substrate (E5), then the generation-speed
track (KV cache -> MTP -> speculation), then the agent feedback
track, with CUDA work waiting for the Linux box.

Status notes: saga E4 (mlx-persistent-tensors) COMPLETED
2026-08-02 -- resident tape, resident optimizer, seam counters,
acceptance met at scale (MLX/CPU 2.96x at d=256, crossover
~d=128; see docs/benchmarks.md). The demo-pedagogy-and-
queue-reorder chore saga (this reorder + the loop-avoidance
demo work) runs before E5.

## Track 0 -- substrate (next saga)

1. **E5 engram-mlx** -- Engram on the E4 TensorHandle seam,
   CONSTRAINED per project-direction: resident
   hashing/addressing, resident gather, resident
   projection+gating, sparse addressed-row identification,
   CPU/MLX parity + perf (use the seam counters), the Tiny-LM
   Engram demo on MLX. Do NOT proceed to E6-E9 (100M-row tables,
   12B retrofit) -- those wait behind the speed and agent tracks.

## Track 1 -- generation speed (the MTP program; CPU + MLX)

Directly after E5. Three separate sagas, each with its own
measurable acceptance -- never one enormous MTP saga.

2. **generation-state-kv-cache** -- the incremental-generation
   engine MTP needs: GenerationState (token ids, position, model
   state, kv cache, rng), begin_generate/generation_step surface,
   per-layer K/V append without prefix recompute, cache
   reset/clone/accounting, CPU + MLX, cache-equivalence tests vs
   full-prefix recompute. Exit: cached generation measurably
   faster, greedy outputs identical cached vs uncached.
3. **mtp-training** -- multi_token_heads / multi_token_loss as
   Model DSL citizens (shifted targets, horizon masks, shared vs
   separate projections, per-horizon loss/accuracy/calibration,
   parameter accounting) + the experiment grid (NTP baseline, MTP
   from scratch, NTP+heads adaptation, frozen-trunk heads-only,
   partial unfreeze, joint).
4. **mtp-self-speculation** -- the first major speed
   demonstration: propose-k / verify-block / accept-prefix /
   repair loop on the KV-cache engine, batched verify_sequence
   primitive, exact-speculative vs greedy semantics kept distinct,
   adaptive proposal depth, full acceptance observability
   (per-horizon acceptance, verifier calls, latency per accepted
   token). Demos: MTP Arithmetic, MTP Code Completion, MTP +
   Engram (do repeated phrases raise far-head acceptance?), MTP
   CPU vs MLX. Exit: real wall-clock win over cached NTP with
   quality-equivalent output under an exact verifier.
5. **speculation-lab** -- the unified proposer harness: a
   TokenProposer trait with Mtp / DraftModel / Ngram / Retrieval /
   Recursive / Hybrid implementations compared under identical
   prompts, target, sampling policy; accepted-tokens/sec,
   time-to-first-token, warm+cold. The heterogeneous topology
   (CPU n-gram -> old CUDA draft -> MLX target) is the long-term
   shape.

## Track 2 -- agent quality (after the first MTP speed result)

6. **agent-episodes-evaluators** -- typed AgentEpisode records
   (task, context, response, tool calls/results, evaluation,
   reward components, critique, outcome) + the Evaluator trait
   (exact output, unit tests, compile/clippy, complexity, file
   constraints, LLM critique, composite) + replay/provenance
   (models, prompts, tool results, evaluator versions, commit,
   seeds, costs). Replaces the stale Saga 18 bundle.
7. **icl-selection** -- demonstration store + selection policies
   (similarity, recency, reward, diversity, contrast pairs,
   error-class match, learned rerank, budget knapsack) +
   deterministic replayable context assembly + compression forms.
   Exit: selection beats random at equal context budget, fully
   inspectable.
8. **icrl-feedback-loop** -- the feedback_loop form over
   episodes+ICL; compare feedback forms (scalar/structured
   reward, critique, failing tests, contrast, lessons); guard
   against false improvement (held-out tasks, independent-retry
   and best-of-N baselines, evaluator-hacking checks). Engram
   enters HERE as one adaptation channel compared against
   ICL/ICRL-history/LoRA -- not as a prerequisite. The headline
   convergence experiment: does speculative speed make
   multi-round ICRL fit one big-model latency budget?

## Track 3 -- mHC (PARKED behind the MTP program; user direction 2026-08-02)

CPU phases parked, accelerator phases deferred. Design source
`docs/mHc-research.txt` + reference impls in
softwarewrighter/mHC-poc.

9. **mhc-p1-constrained-transforms** (parked) -- the reusable
   Sinkhorn-style projection ops (exp/reduce/clamp/div/
   transpose/matmul, all existing primitives).
10. **mhc-p2-cpu-layer** (parked) -- the multi-stream residual
    ModelSpec layer ([batch, tokens, streams, features]) +
    stability demo, CPU-first (the Engram E1->E2 shape).
11. **mhc-p3-mlx-resident** (DEFERRED) -- parity/perf on the
    TensorHandle seam; revisit after the MTP program and the
    parked CPU phases.
12. **mhc-cuda** (DEFERRED to Track 5 hardware).

## Track 4 -- reasoning + adaptive models

13. **recursive-runtime** -- shared recur() primitive (parameter
    sharing across steps, latent state, truncated/full backprop,
    checkpointing, fixed + learned halting, per-step losses,
    state viz) + test-time-compute measurement.
14. **trm-demo** then **hrm-comparison** -- TRM first (forces the
    primitives), HRM as a composition of two recurrent schedules.
15. **bdh-adaptive-state** -- sparse positive activations,
    episode-local dynamic edge state (separate from trained
    params), Hebbian-family local updates, and the
    inspect-and-intervene demo (pick a unit, watch edges change,
    zero an edge, replay, measure the causal delta).
16. **trajectory-tuning** -- LoRA/distillation from recorded agent
    experience (converges Tracks 1+2: does speculative speed make
    multi-round ICRL fit one big-model latency budget?).

## Track 5 -- hardware reach (Linux box required; ALL CUDA work deferred here)

17. **cuda-resident-tensors** -- mlpl-cuda-handle implementing
    DeviceArray/DeviceOps over Candle CUDA tensors (the E4 seam is
    backend-neutral by construction); op bring-up order: matmul,
    elementwise, activations, reductions, softmax/CE, gather,
    optimizer updates, KV append, quantized linear. Includes the
    owed Linux verification of the E3 cuda-target-gating
    manifests + the /mw-cp playbook note. Also retires the
    demoted gpu_step per-shape path (E4 step 010 left it as the
    CUDA route).
18. **engram-cuda + mhc-cuda** (DEFERRED here per user direction
    2026-08-02) -- feature parity on the resident CUDA substrate
    (docs/engram-sagas-plan.md E10 + mHC p4).
19. **quantized-small-models + cpu-fast-path** -- INT8
    weight-only, FP16-where-supported, FP32 fallbacks (no
    FP8/recent-tensor-core requirements); compiled-MLPL CPU track:
    allocation elimination, scratch reuse, SIMD matmul, small-
    matrix threading, mmap'd weights.

## Track 6 -- Colab-replacement gaps (pull-based, ride the tracks)

Driven by whichever research saga needs them first, not as one
monolith: model/data interchange (SafeTensors, HF tokenizer JSON,
checkpoint save/resume incl. optimizer + residency + RNG + dataset
cursor, NumPy import/export, GGUF-read where practical -- E7
already schedules the Llama-family import), experiment sweeps
(cartesian/random, early stop, resumable, report generation), and
the research-tied visualizations (MTP acceptance timelines,
proposal trees, KV-cache size, sync/fallback event panels,
recursive-state trajectories, reward progression).

## Track 7 -- APL2 parity (ML-first ordering; see docs/apl2-parity-gap.md)

20. **apl2-hof-and-order** -- composition + higher-order functions
    in compact ASCII (each, `>>` composition values, `|>` pipe,
    bind, outer/inner, scan) + grade_up/grade_down/sort/compress:
    serves ML (pipelines, top-k, batch ops) and the operator
    algebra at once. Requires first-class user functions (`:u:`).
    Carries the associative-recurrence pedagogy for scan (a time
    loop the language CAN absorb -- see the loop-avoidance demo).
21. **apl2-strings** -- strings as data (split/join/substring/
    find/replace, char codes, formatted output): unlocks ML data
    prep AND the classic report-app class.
22. **apl2-nested-arrays** -- the big one (enclose/disclose,
    depth > 1, each over items; ragged batches for ML); its own
    design saga.
23. **apl2-classic-apps** -- general-purpose completeness gate:
    file I/O, interactive input, dates, formatted output;
    acceptance = the five litmus apps in docs/apl2-parity-gap.md
    actually running (ledger, inventory, menu utility, text
    game, date utility).

## Maintenance items

- **sw-checklist paydown spike** -- the fn-count warning class is
  out of cheap in-step retirements; the dedicated spike + the
  crate-partition work own it, target halving both counts
  (2026-08-02 ledger: 2 documented FAILs, 337 warnings).
- **Wiki errata upkeep** -- the 2026-08-01 audit was resolved on
  2026-08-02 (source docs repaired, resolution table on the wiki
  errata page); the standing rule in CLAUDE.md ("Wiki errata
  discipline") keeps it current from here.
- **Upstream agentrail fix** -- `agentrail instructions apply`
  emits em dashes into the managed CLAUDE.md block; the repo
  markdown gate cannot go fully green until it emits ASCII.

## Explicitly deprioritized / retired

- Engram E6-E9 (sparse 100M tables, checkpoint import at scale,
  12B retrofit headline) wait behind Tracks 1-2.
- Engram/CUDA and mHC/{MLX,CUDA} deferred per user direction
  2026-08-02 (see Tracks 3 and 5).
- The old Saga 18 agent-orchestration bundle is retired in favor
  of items 6-8.
- Capabilities the old roadmap listed as missing but that already
  exist (experiment tracking, datasets, tokenization, autograd,
  Adam/train blocks, MLX resident training, serve/SSE, LoRA,
  Tiny-LM E2E, LLM REST, typed values, model viz, Engram E1-E3)
  do not reappear as milestones.
