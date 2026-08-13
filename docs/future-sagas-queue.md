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

Track 1 is COMPLETE: data-forge (2026-08-04) and
experiment-quality (2026-08-05) both shipped. NEXT SAGA:
generation-state-kv-cache (Track 2). This is the layer every
later track consumes: MTP needs constructed training data,
TRM/HRM need controlled reasoning datasets, ICRL needs validated
task distributions.

2. **data-forge** (COMPLETE 2026-08-04) -- synthetic data as a first-class ML
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
3. **experiment-quality** (COMPLETE 2026-08-05) -- the evaluation
   rigor the research calls for BEFORE drawing MTP/small-model
   conclusions. Shipped: pareto_front / param_count /
   experiment_metric / pareto_plot, the robustness-suite and
   scaffold-present/absent idioms, and the Experiment Quality
   demo category (Robustness Suite, Scaffold Dependence, Pareto
   Frontier). Latency/RAM/energy axes stay externally measured
   until serve telemetry feeds metrics back (design resolution).
   The saga also absorbed the book-gap-coverage program (audit +
   16 glossary entries + 9 demos + 4 diagrams) and seven
   user-reported REPL fixes (colon trichotomy, --help everywhere,
   name-command parsing, usage-table pins).

## Pedagogy completeness (book-coverage; schedulable any time)

**book-gap-coverage** -- close the gaps found by
`docs/book-coverage-audit.md` (the three-eBook completeness check
against glossary/demos/diagrams/lessons): ~16 small glossary
entries (Bayes toolkit, eigen/SVD, Jacobian/Hessian, metrics,
kernel/ensemble umbrellas, segmentation/detection), ~10 demos all
expressible with today's builtins (metrics playground, linear
SVM + kernel trick, voting ensemble, power iteration,
distributions, Bayes grid posterior, augmentation, toy
segmentation, VAE, decision stump), ~4 diagrams (naive Bayes,
kernel trick, Bayes' theorem, eigenvectors), and two findability
text fixes. Docs+demo work, no runtime changes; can interleave
between feature sagas or run as one saga.

**fixed-width-ints-bitops** (added 2026-08-07; demo-memory
request #2, docs/demo-memory-upstream.md) -- fixed-width
unsigned integer views + masks/shifts/popcount/conversions;
unlocks Swiss control bytes, compact Bloom filters, Hamming
indexes, binary sparse retrieval. Largest classical-structures
unlock.

**packed-layouts** (added 2026-08-07; demo-memory request #3)
-- packed/aligned array layouts + observable storage size for
credible bytes-per-key and cache-locality claims. Design-first
(touches the memory model); logical tiny-pointer demos work
before it.

**rng-streams** (added 2026-08-07; demo-memory request #4) --
first-class seeded RNG STATE for composable randomized hashing/
shuffling/workload generation and reproducible substreams.
Non-blocking (seeded randn/sample already deterministic).

**safe-record-lookup** (added 2026-08-07; demo-functional-
pipelines request) -- has_field(record, name) -> 0|1 and
record_get(record, name) -> ok(value)|err(...), so schema
validation stops being exception-driven. Small eval-layer
saga; downstream replaces its blocker fixture with positive
tests. PENDING GO.

**error-messages** (added 2026-08-07; docs/maturation-plan.md
section 1) -- context-aware, fix-suggesting EvalErrors:
did-you-mean for unknown names (edit distance over catalog +
u: space), shape-mismatch errors that print both shapes and
the conflicting axis, name-kind confusion hints, arity errors
that show the signature, and an optional structured
`suggestion` field both humans and tools read. High-leverage
before wider feedback; heavily TDD-able. PRE-FEEDBACK PRIORITY.

**semantic-tooling + --check + catalog-export** (added
2026-08-07; docs/maturation-plan.md section 2) -- the shared
enablers for sw-mlpl-lsp / sw-mlpl-mcp / swml-explain: a
`--check` parse-only flag, a machine-readable catalog export,
and structured access to ASTs / shapes / purity / dependency
graphs / tensor provenance / traces (expose semantics, not
source text). Unblocks the AI-agent-integration program.

**backend-independent-ir** (added 2026-08-07;
docs/maturation-plan.md section 5) -- an IR between MLPL and
its targets; the architectural enabler everything semantic
(explain/optimize/find-optimization) and multi-backend depends
on. Large, foundational, sequence deliberately.

**birds-follow-ups** (added 2026-08-07) -- readable word
wrappers (compose/flip/constant as partial-producing
builtins); the demo-combinators sibling-repo teaching
sequence (per the research file); trains/tacit remain someday.

**COMPLETED 2026-08-07: combinator-birds** (was: NEXT UP after mlplunit round-3 -- apl2-hof completed 2026-08-07; the design step implements Partial per docs/combinators-research.txt) --
"To Mock a Mockingbird"-style coverage of combinators
(identity, kestrel, bluebird/composition, cardinal/flip,
warbler, mockingbird...) as MLPL demos + possibly builtins
over function references; cross-language inspiration:
https://mlajtos.github.io/fluent/?example=combinators .
User research incoming; natural companion to apl2-hof below.

**apl2-hof: each / table / over for u:fns** (added
2026-08-07, user direction) -- now CHEAP thanks to the
callable machinery (UserFnRef + invoke_user_fn_values):
each(f, v) applies a reference per element; table(f, a, b)
is the outer product over a function (BQN's table, APL2's
jot-dot); composition combinators (over/atop) follow the
map_ok precedent. Ships with idioms-doc + demo updates
showing them beside the APL2/BQN spellings. PENDING GO.

**COMPLETED 2026-08-07: mlplunit-round-2 (both halves)** --
global_set + sandboxed fs API + run_script with exit
interception; fixtures proven under their harness (adoption
notes: docs/q-and-a.md 2026-08-07 later). Original entries:

**mlplunit-round-2: in-language-event-reporting** (added
2026-08-07; their section 8) -- explicit global-state escape
hatch (proposed global_set/workspace form; binding hygiene
stays default) so a stateful MLPL reporter sink can count
events; verify variadic print. Assessment: docs/q-and-a.md
2026-08-07. PENDING GO.

**mlplunit-round-2: language-native-runner** (added
2026-08-07; their section 8) -- sandboxed fs API (fs_walk /
read_text / write_text / remove_path, Result-based, lexical
order) generalizing the include FsProvider sandbox, plus
run_script(path, {source_dir, data_dir, capture}) evaluating
a file in a FRESH environment via the chunked include runner
and returning structured status + captured typed test events.
PENDING GO.

**COMPLETED 2026-08-06: emacs-mlpl-mode** (user direction) --
mlpl-mode elisp package: font-lock for the three name kinds +
annotations, run-buffer/run-tests commands consuming the
--test-events JSONL transport, --babel-session integration,
jump-to-test from event rows. sw-MLPL enablers to ship
alongside: a `--check` parse-only flag and a machine-readable
builtin-catalog export. Sketch: docs/q-and-a.md 2026-08-06
late night.

**sudoku-solver-ml-demo** (added 2026-08-06, user direction)
-- a TRAINED model solving Sudoku as an ML demo (candidate
architectures: BDH, HRM, TRM -- hierarchical/tiny recursive
reasoning); builds on the blocks view + the general-programming
backtracking demo as the classical baseline.

**apl2-idioms + expunge** (added AND completed 2026-08-06, user direction)
-- (1) docs/apl2-idioms.mlpl: an EXECUTABLE Rosetta document
mapping APL2 expressions (Unicode comments) to equivalent MLPL
expressions, explicitly marking idioms not yet expressible
(enclose/nesting, each, dyadic iota, ...); loadable in the web
editor and runs clean end-to-end. (2) An APL )ERASE / quad-EX
style expunge builtin to drop a variable or user function
(clear_binding already sweeps every value table; the builtin
is a thin loud wrapper + user-fn removal) -- handy for demos
with unrelated sections. NEXT UP.

**nested-arrays / enclose (APL2 program)** (added 2026-08-06)
-- true nested arrays: enclose/disclose (with axis), each,
depth/match generalization, pervasion rules. Design-first,
multi-saga; the flat-array stopgap (transpose_axes + rank-4
blocked disp) ships in the apl2-blocks saga.

**clustering-demos** (added 2026-08-05, user direction) -- flesh
out the Clustering group beyond K-Means; all expressible with
today's builtins: GMM via EM (softmax responsibilities +
weighted means, with the ellipse-overlay visual), DBSCAN
(pairwise_sqdist + threshold masks + a frontier loop; finds the
moons K-Means cannot), agglomerative at whiteboard scale
(merge-matrix heatmap). Glossary entries ride along (GMM, EM,
DBSCAN, dendrogram).

**math-view** (added 2026-08-05; source analysis in
docs/bqn-sw-mlpl-and-math.txt) -- equations as a DERIVED VIEW of
MLPL programs: math metadata on primitives (printable name,
precedence, notation form, index-expansion rule), expression
extraction from the typed representation, shape-annotated
equations (the teaching feature textbooks omit), composition
expansion, HOF rendering via summation/index notation, and a
DocView IR emitting text/LaTeX/MathML/HTML -- never
source-text-to-LaTeX translation. The @word annotation
namespace (shipped with test metadata) carries the presentation
hints (@math_name / @formula / @doc), harvestable via
annotations(...). SHORT-TERM EXTERNAL PATH, no language work:
org-mode + elisp literate extraction over mlpl source blocks,
made trivial once annotations land.

## Track 2 -- generation speed (the MTP program; CPU + MLX)

4. **generation-state-kv-cache** (CPU cache COMPLETE 2026-08-07: core+controls+bench+demo; mlx-resident-kv deferred as a follow-up).  (NEXT UP -- paused
   2026-08-05 for the mlplunit-unblock program, which closed
   2026-08-06 with every sw-MLPL prerequisite shipped; design
   preserved in docs/kv-cache-design.md, plan archived under
   .agentrail-archive/, resume at gen-state-core) --
   GenerationState, per-layer
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
- **Upstream sw-checklist fix** -- bodiless trait `fn` signatures
  are counted as spanning the FOLLOWING impl block, producing
  phantom Function-LOC warnings (observed on EnvScope,
  2026-08-05: an 11-line impl reported as 36-39 lines).
- **Assessment -> plan reconciliation** (queued 2026-08-08, user
  direction) -- read `docs/sw-mlpl-assessment.txt`, translate its
  recommendations into concrete `docs/plan.md` edits and a set of
  proposed sagas. Do NOT use the `gh` CLI to edit/close issues;
  surface any issue changes to the user to do manually, one at a
  time.

## Assessment-driven track (2026-08-08)

From `docs/sw-mlpl-assessment.txt` (see the roadmap section in
`docs/plan.md`). Ordered; each is its own saga at start.

1. **acceptance-slices** -- finish the two in-flight vertical
   slices: sparse-Safetensors peak-memory measurement
   (demo-ml-utils) and WAV inspect / canonical-copy
   (demo-file-processing). Mostly downstream; upstream only as
   probes surface concrete gaps. Converts "bounded algorithm"
   into MEASURED evidence.
2. **byte-stream-contract** -- a generic `ByteSource` / `ByteSink`
   contract (size / read(offset,count) / write / flush /
   position) for bounded INCREMENTAL I/O, the output/streaming
   half beyond the shipped bounded range reads. Interpreter +
   compiler parity is an acceptance criterion. Biggest
   cross-cutting gap; unblocks GGUF/Safetensors, WAV/MP3/Ogg,
   protocols.
3. **packed-bytes** -- typed `u8` (packed) storage so binary
   workloads make honest bounded-memory claims (bytes are exact
   `f64` today). Pairs with demo-memory `packed-layouts` and the
   stream contract.
4. **compiler-io-parity** -- lower `read_bytes` / `args` / bit
   ops + the byte-stream contract to Rust so standalone CLIs
   compile (hexdump / WAV inspector as standalone binaries). The
   arithmetic + matmul lowering defects are already fixed
   (saga `compiler-apply-binop`).
5. **modules-namespaces** -- qualified names, exports, private
   helpers; fix binding/name-collision hazards in recursive code.
6. **capability-matrix** -- `docs/capabilities.md`: a
   mechanically-checked capability x surface matrix with CAP-*
   ids, seeded from the wiki `Capability-Matrix.md`; downstream
   suites emit machine-readable conformance records a central
   script aggregates.

Later (assessment ranks 5-12, already reflected in the plan
roadmap): GGUF slice, file-processing containers (MP3/ID3/Ogg),
structural sharing / views / builders, UDF fold/scan/unfold,
MLX/CUDA service refactor, speculative decoding, targeted
interpreter perf, web UX polish.

**extension-registry-static-provider** (demo-extensions upstream
contract, new 2026-08-09) -- the recommended FIRST slice of the
demo-extensions native-extension contract, fully scoped in
`docs/companion-demo-extensions.md`. A public host registration +
import path proven via a STATICALLY LINKED provider (sidesteps
dynamic loading, arrays, handles, event loop): (1) a public V1
scalar value/error boundary with contained panics; (2) a public
registration API taking an already-validated descriptor, wired as
a lookup the existing `builtins.rs` chain defers to (NOT a second
dispatch); (3) static-provider registration; (4) a minimal `use`
import loading the `module.mlpl` facade after registration; (5)
help/signature metadata through the existing `mlpl-builtin-catalog`
surface. Acceptance: from a script + REPL, `use hello` then a
call returns typed `i64` 42, a failure is a typed MLPL error, a
panic is contained, `help` shows the signature -- provider
statically linked. Dynamic load / arrays / native handles / event
loop / compiled `--embed-extension` are follow-ups.

Build ONCE with `modules-namespaces` (the public/private `hello`
vs `_hello` split IS qualified-names + private-helpers) and
rationalize the existing static `builtins.rs` dispatch + builtin
catalog rather than duplicating them. `include` (source-text
splicing) stays separate but the `use` keyword must reconcile
with it. Large, architectural -- sequence deliberately.

FIRST SLICE SHIPPED 2026-08-09 (registry-first): the abi +
registry + static `hello` provider + colon-spelling invocation +
help are live (interpreter/REPL/serve). Remaining follow-ups:

- **extensions-use-facade** (with `modules-namespaces`) -- the
  `use hello` construct + dotted `hello.answer()` grammar +
  `module.mlpl` facade (public/private publish). Until this, call
  extensions with the colon spelling `hello:answer()`.
- **extensions-compiler-parity** (after `compiler-io-parity`) --
  a link-time static-provider hook in `mlpl-lower-rs` calling the
  SAME `mlpl_extension_registry::register` at generated-`main`
  startup, so compiled binaries resolve extensions against the
  identical registry with no runtime parser. This is the
  "explicit follow-up contract" the downstream repo requested.
- **extensions-dynamic-load** -- `cdylib`/`dlopen` + ABI-version
  negotiation + the manifest/search-path/trust resolver (A4/A7).
- **extensions-arrays-handles** -- array marshaling + NativeHandle
  values (A3); the demo-extensions "zero-copy/array-lifetime"
  ask lands here.
- **extensions-c-abi-adapter** SHIPPED 2026-08-09 --
  `mlpl-extension-cabi` publishes the canonical `#[repr(C)]` V1
  boundary + `register_c_extension(*const ExtensionDescriptorV1)`,
  which validates a provider's C descriptor, wraps each C invoke
  trampoline in a boxed closure marshaling the scalar set, and
  registers into the safe registry. A statically linked provider's
  descriptor pointer reinterprets byte-for-byte. Scalar values
  only; arrays/handles -> `extensions-arrays-handles`; dynamic
  load -> `extensions-dynamic-load`. (Enabling change: ExtFn is now
  `Arc<dyn Fn>` so a provider can capture its C trampoline.)

**compiler-io-parity phased program** (Saga A shipped 2026-08-09:
compiled value model `CVal` + strings + `write_stdout`/`args`/
`arg` -- first compiled binary that does I/O). Remaining rungs, in
order:

- **compiler-source-loading** (Saga B0) SHIPPED 2026-08-10 --
  `mlpl-build` resolves `include` by running the interpreter's
  `mlpl-source-loader::expand()` (load-once, cycles, sandbox) over
  a filesystem provider, then lowering the flattened AST directly
  (`rt = ::mlpl::__rt`), replacing the raw-text->`mlpl!`-macro
  path. Added `--source-dir`. This is the AST-level front-end the
  later rungs lower from. The EARLIEST demo-file-processing gate is
  cleared; `FnDef` is now the next wall.
- **compiler-functions** (Saga B1) PARAM-ONLY SHIPPED 2026-08-11 --
  `def u:name(params) { body }` lowers to a nested Rust `fn` over
  its parameters (`u:name(args)` routes to it); trailing `return`,
  body-local bindings, and doc-string discard are handled; a
  free/global read is a clear Unsupported error. STILL DEFERRED to
  later rungs: user fns that read globals, control-flow-in-body,
  and Results/records (so functions returning `ok(...)`/records --
  e.g. the rolling-Ridge fit -- are not yet lowerable). Records +
  Results lowering rides with `compiler-control-flow` next.
- **compiler-control-flow** (Saga C) -- lower `if`/`while`/`for`
  (a bounded hexdump loops over chunks). Requires string/CVal
  VARIABLES (bindings that hold CVal, not just DenseArray).
- **compiler-read-bytes** (Saga D) -- lower `read_bytes` (whole +
  `offset,length`) + `file_size` + `write_bytes` to Rust
  (returning/consuming `CVal::Arr` byte arrays). MUST share
  validation + error semantics with the interpreter: compiled
  invalid bytes must be REJECTED not coerced, and runtime write
  errors must propagate not be discarded (a lowered call name
  alone is not acceptance).
- **compiler-process-semantics** (Saga D2) -- lower `print` /
  `eprint` / `exit` / `read_stdin` with clean entry/status
  semantics, and fix the `write_stdout` wrapper appending a
  spurious textual result line after binary stdout.
- **compiler-bit-ops** (Saga E) -- lower band/bor/bxor/shl/shr/
  etc. Then a standalone compiled hexdump / WAV CLI is
  expressible (the demo-file-processing capstone; positive
  byte + format artifact parity + a source-free audit).

**runtime stream handles** (../demo-file-processing second gate,
noted 2026-08-09; see docs/companion-demo-file-processing.md).
Distinct from the compiler track and from in-memory codec
chunking -- a COMPOSITIONAL effects surface:

- **runtime-sink-handle** -- a consumable sink handle: bounded
  writes, partial-write handling, flush/close lifecycle,
  sandboxing, offsets beyond f64 integer ambiguity, and a memory
  high-water invariant (resident memory ~ chunk + writer state,
  not total output). (Path-based `append_bytes` / `write_stdout`
  already ship; this is the persistent-handle generalization.)
- **runtime-source-handle** -- binary stdin + a sequential source
  handle with explicit EOF + backpressure + matching
  error/lifecycle semantics; must reproduce the range-reader
  results across split fields. Enables true stdin-driven
  streaming.

Authorized codec extensions (their third gate) ride the
`extensions-*` track (trust/authorization resolver + dynamic
load); no format-specific builtins upstream.

**codec follow-ups** (../demo-algorithms secondary asks, noted
2026-08-09; the PRIMARY typed-native codec -- `to_native` /
`parse_native`, MLPB header + tag set -- shipped in the
native-codec saga and unblocked that repo). These are the
remaining, lower-priority items:

- **codec-streaming** -- an incremental (chunked) encoder/decoder
  so a large value need not be fully materialized in memory.
- **codec-reference-tables** -- shared-reference / cycle policy
  (dedup repeated sub-values; detect and reject or encode cycles).
- **codec-toml-tagged** -- TOML tagged mode mirroring the JSON
  `$mlpl` tagged-envelope so TOML round-trips non-data value kinds
  (see `docs/serialization-variant-encoding.md`).
- **codec-mlpb-integrity** SHIPPED 2026-08-09 -- MLPB v2 appends a
  CRC32 (u32 LE) over the payload; `to_native` emits v2 and
  `parse_native` verifies it (a corrupted payload that still
  decodes as valid is rejected). Backward compatible on read: v1
  buffers (no checksum) still decode. (`native_integrity.rs`.)
- **codec-migration-hooks** -- version-migration hooks with
  path-aware errors (which field/index failed to decode).
- **codec-numeric-types** -- additional numeric element types at
  the boundary (beyond `f64`), e.g. integer / f32 element arrays.

## GitHub issue reconciliation (manual -- maintainer action)

Per user direction, the agent does NOT edit/close issues via the
`gh` CLI. The assessment's triage of the open issues, for the
maintainer to apply case by case:

- **#7 (interpreter / minimax perf)** -> CLOSE as completed
  (all-MLPL alpha-beta ~20x faster; Rust ttt builtins removed).
  Optionally split narrow follow-ups: oversized host recursion
  stack; reduce temporary DenseArray allocations; first-class
  map/dict if a future demo needs it. Its `#6` dependency is
  stale (`#6` completed 2026-06-02).
- **#8 (connect 3D drops scalars)** -> CLOSE as completed
  (fixed client-side + deployed). Optional smoke: `x = 2 + 3; x`
  reaches the 3D stage.
- **#9 (connect :models/:vars use local state)** -> smoke-test
  HEAD (`:models` / `:vars` / `:wsid` reflect the SERVER
  workspace, per commit ff9ab95 routing fix), then CLOSE.
- **#10 (speculative-decoding demo)** -> CLOSE as superseded by
  the planning docs, or label `research-demo, future,
  non-blocking`.
- **#11 (:ask quoting / modal UX)** -> keep low-priority or move
  the multiline-Ask-dialog feature to web future-work; fold the
  cheap quoting-consistency fix into the next `:ask` touch.

Adopt going forward: saga completion reconciles every referenced
issue, and each issue is one of bug / capability blocker /
forcing-function request / future experiment.

## Downstream companion asks -- algebra text/viz surface

Source: `demo-abstract-algebra/docs/sw-mlpl-work-order.md` (the
downstream repo's ranked upstream asks). Five items shipped and
were adopted downstream: **A1** (bare-filename CLI), **B1**
(`str_concat` / `str_join`), **B2** (`to_string`), **B5**
(string-list as a `u:` argument), and **A4** (the six previously
undocumented `svg()` types are now in `docs/lang-reference.md`).
The saga `algebra-text-surface` closed on those. What remains is
queued below, cheapest first (the downstream doc has the full
symptom / acceptance for each). None is a downstream blocker --
each has an honest workaround in use now.

- **narration-comment-grouping** (D1, feature, one condition) --
  a blank line cannot separate a comment block: `group_statements`
  (`statement_groups.rs`) discards blank lines before flush, so a
  comment block always rides with the following statement. Flush a
  pending comment-only buffer when a blank line is seen. Behavior
  change (splits one group into two) and it makes A3 much less
  reachable. Downstream workaround: a bare `;` after the block.
- **comment-span-render-fix** (A3, bug, small) -- a full-line
  comment in a statement GROUP renders its code inside the italic
  comment span: `split_inline_comment`
  (`components/web-tutorial/crates/mlpl-web-tutorial/src/comment.rs`)
  is fed a whole group and splits at index 0, so `code` is empty
  and later lines vanish into the comment. Split per line, not per
  group. Largely subsumed by D1.
- **viz-table-and-labelled-life** (C1, feature, medium) -- a
  labelled, categorically-coloured grid. Add a `"table"` svg type
  (`svg(t, "table", {row_labels, col_labels, cell_text,
  highlight_rows, highlight_cols})`) plus a categorical /
  labelled / pace-adjustable `life` variant (value selects a
  palette entry, per-frame caption, frame duration as an
  argument). Deletes most of a ~500-line hand-written SVG library
  downstream (Cayley tables). Reference shape: `u:frames_svg` in
  `demo-abstract-algebra/lib/render.mlpl`.
- **recursion-depth-cap** (A2, bug, medium) -- deep `u:` recursion
  aborts the process (stack overflow) instead of a catchable MLPL
  error. Add a recursion-depth cap that raises a normal
  `EvalError` (e.g. "recursion limit (N frames) exceeded in
  u:count"). Robustness.
- **string-ops-len-slice-find-split** (B3, missing, medium) --
  strings have no length / index / search. Add `str_len(s)` ->
  scalar (CHARACTERS, not bytes), `str_slice(s, start, len)` ->
  string (character-indexed), `str_find(s, needle)` -> scalar
  (first index, -1 if absent), `str_split(s, sep)` -> string-list.
  `str_len` counting characters is the load-bearing half.
- **string-list-builders** (B4, missing) -- string lists can only
  arrive as a literal or from `record_keys` / `parse_json` / a
  tokenizer vocab; a program cannot append to one, which also
  makes `str_join` hard to feed. Add `list_append(xs, s)` /
  `list_concat(xs, ys)`, or (smaller) make `concat` accept string
  lists.
- **life-small-board-sizing** (C2, tiny) -- `life` renders small
  boards tiny: `MAX_CELL` (36) binds before the target edge for
  orders 2-8, so a 3x3 board is ~132px. Raise `MAX_CELL` (~80) or
  add an optional size argument
  (`components/viz/crates/mlpl-viz-marks/src/life.rs`). One
  constant.
- **viz-digraph-type** (C3, feature) -- no node-link diagram type.
  Add a `"digraph"` type taking an `[N, N]` adjacency matrix with
  node labels via `aux`, laying nodes on a circle. Also gives
  `knn_graph` a renderer.
- **block-comment-syntax** (D2, feature) -- no block comment form;
  narration is runs of `#` lines. Add `#* ... *#` (fits the
  existing `#` lexeme, unambiguous with the line form).
- **source-narration-kind** (D3, feature) -- a pasted / uploaded
  file cannot declare `EntryKind::Narration` (intro / takeaway
  prose with no prompt); only the catalog runner / upload / status
  paths create one. Promote a leading comment block to the intro
  and a trailing block to the takeaway.

## Explicitly deprioritized / retired

- Engram E6-E9 wait behind Tracks 1-3.
- Engram/CUDA and mHC/{MLX,CUDA} deferred (Tracks 4/6).
- The old Saga 18 bundle is retired in favor of items 9-14.
- Shipped capabilities do not reappear as milestones.
