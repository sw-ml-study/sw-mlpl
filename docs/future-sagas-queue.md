# Future sagas queue

REPRIORITIZED 2026-08-02 (revision B) per the "==== 20260802"
research section of `docs/project-direction.txt`. That analysis
surveyed the Software Wrighter Lab playlist corpus against the
planned scope and concluded the roadmap was still too
model-feature oriented: the most consequential missing layer is
HOW LEARNING EXPERIENCES ARE CREATED, VALIDATED, SELECTED,
SEQUENCED, EVALUATED, AND REUSED. Data-forge, rejection
sampling, curriculum, and distillation are therefore elevated to
the same strategic level as MTP and ICRL.

The four-priority strategy is unchanged (generation speed, agent
quality, Colab replacement, small/old hardware). Each numbered
item becomes its own agentrail saga with its own plan at start.

> **mHC placement note:** the 20260802 research suggests the mHC
> CPU+MLX bounded track directly before the generation-state
> work. The user's explicit same-day direction ("defer
> mHC/{CUDA,MLX}, park mHC-CPU behind MTP") takes precedence
> until countermanded, so mHC remains parked in Track 4 below.

Status notes: saga E4 (mlx-persistent-tensors) completed
2026-08-02 and saga E5 (engram-mlx) completed 2026-08-03 --
resident tape + resident engram, crossover ~d=128, one CPU
fallback per step (docs/benchmarks.md). The next saga is
data-forge (Track 1).

## Track 0 -- substrate (COMPLETE)

1. **E5 engram-mlx** (COMPLETE 2026-08-03) -- Engram trains fully
   resident on the E4 seam (one CPU fallback per step: fused CE
   backward); selection-matmul retained as the resident gather
   (exact scatter-ADD backward); dev concat/split landed;
   crossover ~d=128 like the base tiny-LM (benchmarks.md E5
   section). E6-E9 remain deferred. NEXT SAGA: data-forge
   (Track 1).

## Track 1 -- learning-experience infrastructure (VERY HIGH; new 20260802)

Directly after E5. This is the layer every later track consumes:
MTP needs constructed training data, TRM/HRM need controlled
reasoning datasets, ICRL needs validated task distributions.

2. **data-forge** -- synthetic data as a first-class ML
   algorithm: generators (template, grammar, knowledge-graph,
   teacher-model, mutation/adversarial), candidate evaluators
   (exact oracle, graph-path verifier, compiler/tests, reward
   model, LLM judge), the generic rejection-sampling/ranking
   substrate (sample_many / evaluate_many / select_best /
   filter_reward -- reused later by MTP proposal analysis, ICRL,
   distillation, recursive-model voting), curation (dedupe,
   balance, difficulty stratification, curriculum schedules),
   and dataset provenance. Includes the MINIMAL knowledge-graph
   task infrastructure (entities/relations, path enumeration,
   query generation, answer verification, graph splits) -- full
   GNNs explicitly out of scope. Demo curricula: arithmetic,
   graph multi-hop, Rust repair, tool use.
3. **experiment-quality** -- the evaluation rigor the research
   calls for BEFORE drawing MTP/small-model conclusions:
   distribution-shift and prompt-format robustness suites
   (paraphrase, field reorder, irrelevant context, identifier
   changes, hop extension, scaffold present/absent) plus
   Pareto/efficient-frontier analysis as a native experiment
   concept (quality vs parameters / bytes / peak RAM / latency /
   energy; acceptance rate vs MTP overhead), rendered via the
   existing svg/experiment machinery.

## Track 2 -- generation speed (the MTP program; CPU + MLX)

4. **generation-state-kv-cache** -- GenerationState, per-layer
   K/V append without prefix recompute, cache
   reset/clone/accounting, batched verification hooks, CPU +
   MLX, cache-equivalence tests. Exit: cached generation
   measurably faster, greedy outputs identical.
5. **mtp-training** -- multi_token_heads / multi_token_loss as
   Model DSL citizens + the experiment grid (NTP baseline, MTP
   from scratch, adaptation variants) -- trained on data-forge
   curricula, evaluated through experiment-quality suites.
6. **mtp-self-speculation** -- propose-k / verify-block /
   accept-prefix / repair on the KV engine; exact-speculative vs
   greedy kept distinct; full acceptance observability. Exit:
   real wall-clock win with quality-equivalent output under an
   exact verifier.
7. **quantization-reference** (new 20260802; promoted from the
   deferred-QLoRA line) -- correct quantization SEMANTICS before
   kernels: symmetric/asymmetric, per-tensor vs per-channel,
   static vs dynamic activations, calibration data, error
   histograms, layer sensitivity, fake-quant + dequantized
   reference execution. Enables draft models, larger imported
   students, and cascaded speculation. Optimized kernels stay in
   Track 6; QLoRA becomes an application of this representation.
8. **speculation-lab** -- the unified TokenProposer harness (Mtp
   / DraftModel / Ngram / Retrieval / Recursive / Hybrid)
   compared under identical prompts/target/sampling;
   accepted-tokens/sec, time-to-first-token, warm+cold.

## Track 3 -- agent quality (feedback program)

9. **agent-episodes-evaluators** -- typed AgentEpisode records +
   the Evaluator trait + replay/provenance (shares evaluator
   machinery with data-forge).
10. **icl-selection** -- demonstration store + selection
    policies + skill libraries + deterministic replayable
    context assembly. Exit: selection beats random at equal
    context budget.
11. **scaffolded-reasoning** (new 20260802) -- privileged
    structure during training, removed at inference: scaffold
    types (graph paths, execution traces, compiler diagnostics,
    tool-call traces, decomposition hints), experiments (full /
    partial / dropout / curriculum removal / no-scaffold eval),
    distribution matching between generated training tasks and
    evaluation tasks. The primary small-model reasoning-quality
    demo; consumes data-forge graphs and curricula.
12. **icrl-feedback-loop** -- feedback_loop over episodes+ICL;
    guard against false improvement (held-out tasks, best-of-N
    baselines, evaluator-hacking checks). Engram enters here as
    one adaptation channel. Headline experiment: does
    speculative speed make multi-round ICRL fit one big-model
    latency budget?
13. **distillation-core** (new 20260802; unbundled from old Saga
    18) -- teacher-to-student transfer as a research facility:
    logit KL, sequence, feature, policy/trajectory,
    self-distillation (including MTP heads and draft models),
    and distillation-with-rejection-sampling (Rust compile/tests
    as the cheap verifier). Precedes large imported-model work
    because it gives imports an adaptation path.
14. **trajectory-tuning** -- LoRA/distillation from recorded
    agent experience (converges Tracks 2+3).

## Track 4 -- reasoning + adaptive models

15. **recursive-runtime** -- shared recur() primitive +
    test-time-compute measurement; build the generic
    ACTIVATION-CAPTURE SEAM here (per 20260802 item 10) so
    interpretability is not BDH-exclusive.
16. **trm-demo** then **hrm-comparison**.
17. **causal-inspection** (new 20260802) -- intervention, not
    just observation: activation capture/replacement/zeroing,
    neuron/edge ablation, activation patching, linear probes,
    causal tracing, gradient attribution; applied to BDH, mHC
    streams, MTP heads, Engram gates, TRM recursion, and
    teacher/student comparisons.
18. **bdh-adaptive-state** -- sparse activations, episode-local
    edge state, Hebbian updates, inspect-and-intervene demo (on
    the causal-inspection seam).
19. **mHC bounded track** (PARKED here per user direction
    2026-08-02; see the placement note above) --
    p1-constrained-transforms, p2-cpu-layer;
    p3-mlx-resident and mhc-cuda remain DEFERRED.

## Track 5 -- efficiency + architecture (small/old hardware)

20. **small-model-architecture-lab** (new 20260802) -- the
    MobileLLM-style ablation lab at fixed parameter count:
    deep-narrow vs shallow-wide, weight tying, GQA/shared-KV,
    factorized embeddings, bottleneck FFNs, layer sharing,
    early exit, recurrent depth -- measured on the
    experiment-quality Pareto axes (quality, CPU/MLX latency,
    memory, serialized size). More useful than five disconnected
    architecture demos; feeds TRM/HRM/mHC comparisons.
21. **associative-energy-models** (medium-low) -- Hopfield-family
    associative memory as the Engram complement (hash-addressed
    vs content-addressed vs attractor recall), energy-based
    candidate ranking, energy minimization as test-time compute.
    After recursive-runtime; educational value.
22. **latent-world-model** (later) -- JEPA-style latent
    prediction on grid worlds / cellular automata / board
    states (no vision stack); potential substrate for agent
    world models and ICRL state abstraction. Must not displace
    anything above.

## Track 6 -- hardware reach (Linux box required; ALL CUDA work deferred here)

23. **cuda-resident-tensors** -- mlpl-cuda-handle over Candle
    CUDA tensors; includes the owed Linux verification of the
    cuda-target-gating manifests and retiring the demoted
    gpu_step per-shape path.
24. **engram-cuda + mhc-cuda** (deferred per user direction).
25. **optimized-quantized-kernels + cpu-fast-path** -- the
    kernel side of quantization-reference: INT8/INT4 weight-only
    CPU kernels, MLX-compatible representations, CUDA fallbacks;
    compiled-MLPL CPU track (allocation elimination, SIMD
    matmul, mmap'd weights).

## Track 7 -- Colab-replacement gaps (pull-based, ride the tracks)

Model/data interchange (SafeTensors, HF tokenizer JSON,
checkpoint save/resume incl. optimizer + residency + RNG,
NumPy import/export, GGUF-read where practical), experiment
sweeps (cartesian/random, early stop, resumable, reports), and
research-tied visualizations (MTP acceptance timelines, KV-cache
size, sync/fallback panels, reward progression). data-forge and
experiment-quality now own the dataset-generation and
robustness/report slices of this track.

## Track 8 -- APL2 parity (ML-first ordering; see docs/apl2-parity-gap.md)

26. **apl2-hof-and-order** -- each / `>>` / `|>` / bind /
    outer / inner / scan + grade/sort/compress; requires
    first-class user functions; carries the
    associative-recurrence pedagogy for scan.
27. **apl2-strings**, 28. **apl2-nested-arrays**,
29. **apl2-classic-apps** -- unchanged scope and order.

## Explicit non-goals (20260802 research)

Playlist subjects that should exercise reusable capabilities,
not become tracks: individual 1B/3B model families, any single
mobile model, benchmark leaderboards, production-scale Boltzmann
machines, a full JEPA vision stack, and generic RAG before the
retrieval/evaluation problem is defined (knowledge-graph tasks
in data-forge come first).

## Maintenance items

- **sw-checklist paydown spike** -- dedicated spike + crate
  partitions own the fn-count class (2026-08-02 ledger: 2
  documented FAILs, 337 warnings).
- **Wiki errata upkeep** -- standing rule in CLAUDE.md.
- **Upstream agentrail fix** -- `agentrail instructions apply`
  emits em dashes; the repo markdown gate needs ASCII.

## Explicitly deprioritized / retired

- Engram E6-E9 wait behind Tracks 1-3.
- Engram/CUDA and mHC/{MLX,CUDA} deferred (Tracks 4/6).
- The old Saga 18 bundle is retired in favor of items 9-14.
- Shipped capabilities do not reappear as milestones.
