# Gaps To Be Addressed

**Status:** planning doc, 2026-05-04. Tracks ML concepts that
appear in the v0.19 glossary (`docs/glossary.md`) and the
145-term ML reference list (Software Wrighter
`blog-planning/docs/ml-terms-alphabetized.txt`) but are NOT
shipped as MLPL features or demos today. Each row points at
the saga or follow-up that would close the gap, or notes
that the concept is intentionally out of MLPL's scope.

Use this doc when planning future sagas or vetting
"comprehensive ML coverage" claims. Update it whenever a
glossary entry's status changes from "deferred" to "shipped".

## Legend

| Marker     | Meaning                                                       |
|------------|---------------------------------------------------------------|
| `SCOPE`    | Out of MLPL's teaching-language scope (deployment, MLOps)     |
| `CORE`     | Core ML concept worth a builtin or a demo eventually          |
| `LLM`      | Modern-LLM concept; lands when MLPL grows the LLM surface     |
| `RESEARCH` | Research-grade idea; folds into a future research saga        |
| `META`     | Cross-cutting / philosophical; better as a guide than builtin |

## Coverage table

### Architecture and layer families

| Concept                         | Marker   | Closest saga / note                                      |
|---------------------------------|----------|----------------------------------------------------------|
| Convolution / CNN               | CORE     | Conv-family saga (post-26); image demos depend on it     |
| RNN / LSTM / GRU                | CORE     | Recurrent-family saga; mostly historical pedagogy        |
| State Space Models / Mamba      | RESEARCH | Linear-recurrence saga; probably after Sagas 24-28       |
| MoE (Mixture of Experts)        | LLM      | Sparse-arch saga; depends on routing primitives          |
| Sparse Activation               | RESEARCH | Pairs with MoE / lottery-ticket saga                     |
| GQA / MQA                       | LLM      | Attention-variant saga; small follow-up to Saga 11       |
| Flash Attention                 | LLM      | Kernel-fusion follow-up to Saga 14 (MLX)                 |
| RoPE                            | LLM      | Positional-encoding-variants saga                        |
| MLA (Multi-head Latent Attn.)   | LLM      | Attention-variant saga                                   |
| Diffusion Models                | LLM      | Generative-model saga (post-Saga 24 distributions)       |
| Autoencoder                     | CORE     | Pairs with VAE under Saga 24                             |
| VAE (Variational Autoencoder)   | CORE     | Saga 24 ships Distribution; VAE demo follows             |
| GAN                             | CORE     | Generative-model saga; after Distributions               |
| VLM (Vision-Language Models)    | LLM      | Vision saga; out of v0.19 scope                          |
| Perceptron                      | META     | Historical context; could be a one-line demo             |

### Loss / training mechanics

| Concept                         | Marker   | Closest saga / note                                      |
|---------------------------------|----------|----------------------------------------------------------|
| Dropout                         | CORE     | Regularization saga; small builtin                       |
| Batch Normalization             | CORE     | Norm-variants saga; follow-up to RMS norm                |
| Layer norm                      | CORE     | Same saga as Batch norm                                  |
| Weight Decay                    | CORE     | Pairs with `adam` upgrade to AdamW                       |
| Weight Initialization (Xavier/He)| CORE    | Init-strategies saga; small enum on `linear`             |
| Gradient Clipping               | CORE    | Optimizer-utility saga                                   |
| Label Smoothing                 | CORE    | Loss-utility saga                                        |
| Mixed Precision (f16/bf16)      | CORE    | Dtype layer; deferred indefinitely under v0.19 ranking   |
| Quantization                    | CORE    | QLoRA follow-up to Saga 15                               |
| Stop gradient / detach          | CORE    | Autograd-utility saga; small builtin                     |
| Checkpointing (state save/load) | CORE    | Persistence saga; uses safetensors-style format          |
| Early Stopping                  | CORE    | Train-loop control saga; condition on `last_losses`      |
| KV Cache                        | LLM     | Generation-efficiency saga; follow-up to Saga 13         |
| Speculative Decoding            | LLM     | Generation-efficiency saga                               |
| Beam search                     | LLM     | Decoding-strategy saga                                   |

### Evaluation, calibration, and reliability

| Concept                         | Marker   | Closest saga / note                                      |
|---------------------------------|----------|----------------------------------------------------------|
| Precision vs Recall / F1        | CORE     | Eval-builtins saga; small builtins                       |
| ROC / AUC                       | CORE     | Eval-builtins saga                                       |
| Calibration / Miscalibration    | CORE     | Eval-builtins saga; temperature scaling fits naturally   |
| Uncertainty Estimation          | RESEARCH | Saga 20 ensembling is a tiny taste; Bayesian deferred    |
| Cross-Validation                | CORE     | Dataset-utility saga                                     |
| Test set                        | CORE     | Convention; partial via `val_split`. Add `test_split`    |
| Confusion matrix                | -        | Already shipped; no gap                                  |

### Distribution shift, robustness, generalization

| Concept                         | Marker   | Closest saga / note                                      |
|---------------------------------|----------|----------------------------------------------------------|
| Distribution Shift / Covariate Shift | CORE | Eval-robustness saga                                  |
| Concept Drift / Data Drift      | SCOPE    | MLOps; out of scope                                      |
| OOD Inputs                      | CORE     | Robustness saga                                          |
| Adversarial Examples            | RESEARCH | Robustness / safety saga                                 |
| Shortcut Learning / Spurious Correlations | META | Documentation; could be a demo                       |
| Data Augmentation               | CORE     | Dataset-utility saga                                     |
| Data Leakage                    | META     | Documentation; lint-style check possible                 |
| Curriculum Learning             | RESEARCH | Train-loop saga                                          |

### Theory and interpretability

| Concept                         | Marker   | Closest saga / note                                      |
|---------------------------------|----------|----------------------------------------------------------|
| Bias-Variance Tradeoff          | META     | Documentation; one-page guide                            |
| Curse of Dimensionality         | META     | Documentation                                            |
| Universal Approximation         | META     | Documentation                                            |
| Inductive Bias                  | META     | Documentation                                            |
| Manifold Hypothesis             | META     | Documentation                                            |
| Loss Landscape / Sharpness      | RESEARCH | Visualization saga                                       |
| Lottery Ticket Hypothesis       | RESEARCH | Pruning saga                                             |
| Double Descent                  | RESEARCH | Could be a small interactive demo                        |
| Grokking                        | RESEARCH | Demo-friendly; small modular-arithmetic example          |
| Optimization vs Generalization  | META     | Documentation                                            |
| Mechanistic Interpretability / Superposition | RESEARCH | Probing saga; depends on graph-introspection (Saga 25) |
| Neural Collapse                 | RESEARCH | Visualization saga                                       |

### LLM / agentic concepts

| Concept                         | Marker   | Closest saga / note                                      |
|---------------------------------|----------|----------------------------------------------------------|
| GPT (architecture name)         | -        | "Tiny LM" demos already exemplify; documentation only    |
| BERT (architecture name)        | LLM      | Encoder-only LM saga; small follow-up                    |
| Pretraining + Fine-tuning       | -        | Already covered (Saga 13 + Saga 15); no gap              |
| LoRA                            | -        | Already shipped (Saga 15)                                |
| Distillation                    | LLM      | Saga 18 plans this                                       |
| RLHF / DPO / RSFT               | LLM      | Preference-learning saga (Saga 18)                       |
| Constitutional AI               | LLM      | Layered on RLHF saga                                     |
| Reward Hacking / Goodhart       | META     | Documentation; appears in RLHF saga                      |
| Hallucination                   | LLM      | RAG saga (combines `llm_call` + `knn`)                   |
| RAG (Retrieval-Augmented Gen.)  | LLM      | RAG saga; combines `pairwise_sqdist` / `knn` / `llm_call`|
| Chain of Thought                | LLM      | Prompting pattern; documentation, not a builtin          |
| Few-shot / In-Context Learning  | LLM      | Prompting pattern; documentation                         |
| Prompting / Prompt Injection    | META     | Documentation in `using-llm-tool.md`                     |
| Tool Use                        | LLM      | Tool-protocol saga; extends `llm_call`                   |
| Jailbreaks                      | META     | Documentation; safety chapter                            |
| Context Window                  | -        | Implicit in Saga 13 demos; documentation                 |
| Scaling Laws / Compute-Optimality| META    | Documentation                                            |
| Emergent Behavior               | META     | Documentation                                            |
| Latent Space                    | -        | Saga 16 visualization is a tiny taste; no major gap      |

### Safety, alignment, ops

| Concept                         | Marker   | Closest saga / note                                      |
|---------------------------------|----------|----------------------------------------------------------|
| Catastrophic Forgetting         | CORE     | Documentation + LoRA pattern (Saga 15)                   |
| Elastic Weight Consolidation    | RESEARCH | Continual-learning saga                                  |
| Replay Buffers                  | RESEARCH | RL saga                                                  |
| Memory-Augmented Networks       | RESEARCH | Engram saga (Saga 18)                                    |
| Model Editing / Steerability    | RESEARCH | Out of v0.19 scope                                       |
| A/B Testing Models              | SCOPE    | MLOps                                                    |
| Blue/Green / Canary / Shadow    | SCOPE    | MLOps                                                    |
| Production Rollbacks            | SCOPE    | MLOps                                                    |
| Caching Strategies              | SCOPE    | MLOps                                                    |
| Cold Start Problems             | SCOPE    | MLOps                                                    |
| Cost vs Quality Tradeoffs       | SCOPE    | MLOps                                                    |
| Failure Analysis                | SCOPE    | MLOps                                                    |
| Inference Parallelism           | SCOPE    | MLOps                                                    |
| Latency vs Throughput           | SCOPE    | MLOps                                                    |
| Monitoring / Drift Detection    | SCOPE    | MLOps                                                    |
| System Reliability vs Quality   | SCOPE    | MLOps / META                                             |
| Benchmark Leakage               | SCOPE    | MLOps / META                                             |

### Sequence / generation utilities

| Concept                         | Marker   | Closest saga / note                                      |
|---------------------------------|----------|----------------------------------------------------------|
| Beam search                     | LLM      | Decoding-strategy saga                                   |
| KV Cache                        | LLM      | Generation-efficiency saga                               |
| Speculative Decoding            | LLM      | Generation-efficiency saga                               |
| Perplexity                      | LLM      | One-line composition: `exp(cross_entropy(...))`. Add as a builtin in an eval saga |
| Padding                         | -        | Already covered (`batch` returns `batch_mask`)           |
| Tokenization                    | -        | Already covered (Saga 12)                                |

### Other gaps to track

| Concept                         | Marker   | Closest saga / note                                      |
|---------------------------------|----------|----------------------------------------------------------|
| Sinkhorn normalization          | RESEARCH | Norm-variants saga; mentioned in research3.txt           |
| Ensembling                      | -        | Already shipped (Saga 20 Neural Thicket)                 |
| Conditional Computation         | RESEARCH | Pairs with MoE saga                                      |
| Parameter Routing               | RESEARCH | Pairs with MoE saga                                      |
| Self-Training Instability       | RESEARCH | Distillation saga (Saga 18)                              |
| SAM (Sharpness-Aware Min.)      | RESEARCH | Optimizer-variants saga                                  |
| Why Interpretability Is Hard    | META     | Documentation                                            |
| Why ML Is Fragile               | META     | Documentation                                            |
| Why More Data Beats Better Models | META   | Documentation                                            |

## Suggested follow-up sagas (rough grouping)

These are clusters of related gaps that could ship as a single
saga rather than scattered one-offs. None are committed; this
is planning material only.

### Saga "Norm + Init" (CORE)
Batch norm, Layer norm proper, Weight decay (AdamW), Xavier /
He init. Small surface, big learning value -- every demo
benefits.

### Saga "Eval Suite" (CORE)
Precision / Recall / F1, ROC / AUC, Calibration + temperature
scaling, perplexity as a builtin. Tied together by a common
"how good is this model" theme.

### Saga "Sequence Decoding" (LLM)
Beam search, KV cache, top-p sampling, speculative decoding.
A natural follow-up to Saga 13's Tiny LM.

### Saga "Robustness / Distribution Shift" (CORE)
Distribution-shift simulation, OOD detection, adversarial
input attacks. Includes data-augmentation builtins.

### Saga "Conv Family" (CORE)
`conv2d`, `pool`, image-data demos, basic CNN architecture.
Opens a whole new lesson track.

### Saga "RAG" (LLM)
Retrieval-augmented generation pipeline using
`pairwise_sqdist`, `knn`, `llm_call`. Touches Saga 16 and
Saga 19. Also unlocks "hallucination mitigation" demo.

### Saga "Train Loop Polish" (CORE)
Early stopping, gradient clipping, label smoothing,
checkpoint save/load. Small-but-frequent quality-of-life
items currently missing from `train { ... }`.

### Saga "Conv2d + Image Demos" (CORE)
See Conv Family above; explicitly call out for image
classifiers and CNN demos.

### Saga "RNN family" (CORE)
Mostly pedagogical; tiny demo of vanishing-gradient pain
that motivates attention.

## See also

- `docs/glossary.md` -- the term-by-term definitions this doc
  cross-references.
- `docs/optional-typing-design.md` -- the umbrella design for
  Sagas 23-28.
- `docs/saga.md` -- shipped + planned saga history.
- `/Users/mike/github/softwarewrighter/blog-planning/docs/ml-terms-alphabetized.txt`
  -- the 145-term reference list.
