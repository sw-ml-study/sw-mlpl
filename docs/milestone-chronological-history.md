# Chronological-history learning track

Proposed milestone (number TBD, awaiting user approval).

## Why this exists

MLPL today has two single-lesson summaries of ML history --
`HISTORY_OF_ML` (architectures, Perceptron through Transformer) and
`HOW_MODELS_LEARN` (training paradigms). They are good as
orientation lessons but they paint the whole arc in three
paragraphs each. A learner who wants to spend an hour on
*backpropagation* specifically -- where it came from, what problem
it solved, what the Rumelhart-Hinton-Williams 1986 paper actually
proposed -- has no per-concept depth.

This milestone factors out one focused lesson per
paper-introduced concept and adds a new learning path that walks
them in publication order. Other paths can recombine the same
lesson set in different orderings (by architecture family, by
training paradigm, by application domain) without duplicating the
content.

## Goals

- **Educational.** Each lesson is one concept, one foundational
  paper, one runnable MLPL example (where the language can express
  it) or a clear "deferred" note (where it cannot).
- **Composable.** Lessons are the unit. Paths are how they are
  composed. A new path needs zero new content -- just an ordered
  list of existing lessons.
- **Pace-friendly.** A learner can spend 20 minutes on one
  lesson and come back next session. The new path makes that
  navigable.
- **Honest about the boundary.** MLPL covers Perceptron / MLP /
  attention / Transformer; CNN, RNN, LSTM, Diffusion, RLHF etc.
  are glossary-only today. The history lessons say so explicitly.

## Non-goals

- **A textbook.** Each lesson is 200-400 words, not 2000.
- **All of ML history.** The track covers the spine -- the papers
  that introduced new architectural or training ideas that are
  still load-bearing. Empirical papers (BERT-large, GPT-3, etc.)
  matter for the field but not for *concept* introduction.
- **Live-running every era.** Where MLPL has the primitives
  (Perceptron, MLP, Backprop, Attention, Transformer), the lesson
  ships runnable code. Where it does not (CNN, RNN, LSTM,
  Diffusion), the lesson points at the glossary entry and the
  saga note for what would be needed.

## Per-lesson template

Every history lesson follows the same structure:

```
title:    "<technique>: <year> -- <one-sentence framing>"
intro:    Paragraph 1 -- the problem that motivated the paper.
          Paragraph 2 -- the key idea the paper introduced.
          Paragraph 3 -- what came before, what came next.
          Paragraph 4 -- "Working in MLPL today" OR "Deferred:
          here is why."
examples: 3-5 runnable lines IF MLPL has the primitives.
          For deferred eras, a one-line glossary reference instead.
try_it:   One concrete experiment the reader can run or read.
```

Every lesson cites the foundational paper with author + year (no
URLs, per the project's no-external-link policy).

## Initial lesson set (proposed, in chronological order)

1. **McCulloch-Pitts neuron (1943)** -- the first formal model of
   a "neuron" as a thresholded sum. MLPL: build one by hand from
   `linear` + a hard step. Deferred: actual hard-step
   non-differentiable activation.

2. **Hebbian learning (1949)** -- "neurons that fire together,
   wire together." Glossary-only; the rule is an unsupervised
   weight-update that is not in MLPL's optimizer set.

3. **Perceptron (Rosenblatt 1958)** -- the first machine that
   *learned* from data. MLPL: `linear(2, 1, seed)` + `tanh_layer`
   trained on a simple separable dataset; XOR fails (the canonical
   demonstration).

4. **Backpropagation (Rumelhart, Hinton, Williams 1986)** --
   training multi-layer networks via the chain rule. MLPL ships
   reverse-mode autograd, so this is the lesson where `grad(loss,
   wrt)` finally makes sense from first principles. Includes a
   side-by-side: hand-rolled chain rule vs `grad`.

5. **LeNet / CNN (LeCun 1989)** -- weight-sharing and translation
   equivariance for images. Deferred in MLPL: no `conv2d`
   primitive yet; the lesson explains why CNNs mattered and what
   adding `conv2d` would require.

6. **Universal approximation theorem (Cybenko 1989, Hornik 1991)**
   -- the math result that says an MLP with one hidden layer can
   approximate any continuous function. Lesson is conceptual
   (not paper-specific); MLPL example: a 2-input MLP approximating
   `sin(x+y)` on a grid.

7. **LSTM (Hochreiter and Schmidhuber 1997)** -- gates that
   prevent gradient vanishing in recurrent networks. Deferred in
   MLPL: no recurrent layer; lesson explains the gating idea and
   what an LSTM cell would compile to.

8. **Word2Vec / dense word embeddings (Mikolov et al. 2013)** --
   words as vectors in continuous space; the
   "king - man + woman = queen" canonical example. MLPL ships
   `embed`, so the lesson uses it on a tiny corpus.

9. **AlexNet (Krizhevsky, Sutskever, Hinton 2012)** -- the GPU
   moment. Conceptual lesson on scale: AlexNet was not a
   fundamentally new architecture, it was a CNN with more
   compute. MLPL ships the MLX backend; the lesson connects
   "more compute" to today's tradeoffs.

10. **Adam optimizer (Kingma, Ba 2014)** -- per-parameter
    learning rates via first and second moment estimates. MLPL:
    every training demo uses `adam`; the lesson explains the
    moments + bias correction.

11. **Dropout (Srivastava et al. 2014)** -- regularization by
    randomly zeroing activations during training. Deferred in
    MLPL: no `dropout` builtin (it would be a stochastic layer).
    Lesson explains why a stochastic layer is harder to fit into
    the language's eager / deterministic surface.

12. **Batch Normalization (Ioffe, Szegedy 2015)** -- normalize
    layer inputs to stabilize training. Deferred in MLPL: no
    `batch_norm`; the rms_norm primitive is the closest
    living analog. Lesson explains the math and the difference.

13. **ResNet (He et al. 2015)** -- skip connections let gradients
    flow past many layers. MLPL: `residual(block)` ships; the
    lesson uses it.

14. **Attention is All You Need (Vaswani et al. 2017)** -- the
    Transformer paper. MLPL: existing demos cover this in depth;
    the lesson is the "why" framing.

15. **GANs (Goodfellow et al. 2014)** -- two-network adversarial
    training. Deferred: lesson explains the generator/discriminator
    dance and what `gan` plumbing would need.

16. **VAE (Kingma, Welling 2013)** -- variational autoencoders;
    a probabilistic generative model with a tractable latent.
    Deferred: lesson explains the reparameterization trick and
    what `kl_div` + sample-from-gaussian builtins would unlock.

17. **GPT-1 (Radford et al. 2018)** -- the first decoder-only
    transformer LM at scale. MLPL: existing Tiny LM demos are
    pedagogically equivalent at smaller scale.

18. **BERT (Devlin et al. 2018)** -- masked-LM pretraining for
    bidirectional encoders. Lesson contrasts decoder-only vs
    encoder-only objectives; MLPL has the building blocks but no
    masking objective surface.

19. **Vision Transformer (Dosovitskiy et al. 2020)** -- patches
    as tokens. MLPL: just shipped (Saga 29). The lesson is the
    framing piece for what the user has already seen in code.

20. **CLIP (Radford et al. 2021)** -- dual-encoder image-text
    contrastive learning. Glossary entry exists; demo deferred.

21. **InstructGPT / RLHF (Ouyang et al. 2022)** -- preference
    learning over (chosen, rejected) pairs. Deferred; existing
    `HOW_MODELS_LEARN` lesson covers the framing.

22. **LoRA (Hu et al. 2021)** -- low-rank adapters for cheap
    fine-tuning. Glossary entry exists; demo deferred.

23. **Mixture of Experts (Shazeer et al. 2017, Fedus et al. 2022)**
    -- sparse routed FFN. Glossary entry exists; demo deferred.

24. **Mamba / State-space models (Gu, Dao 2023)** -- alternative
    to attention. Glossary entry exists; demo deferred.

Total: 24 focused lessons. About half are runnable, about half are
"deferred + here is why."

## Chronological learning path

`apps/mlpl-web/src/paths.rs` gets a new entry:

```
LearningPath {
    title: "The chronological history of ML",
    blurb: "Every major ML paper that introduced a new concept,
            in publication order. Half the lessons ship runnable
            MLPL code; the other half are framing pieces for
            techniques that MLPL has not implemented yet.",
    steps: &[
        // One Step::Lesson per lesson, plus matching diagrams +
        // glossary entries interleaved. Plus Step::Note bookends.
    ],
}
```

## Other paths reusing the same lessons

The lessons live in `lessons.rs` / `lessons_advanced.rs`; the
path is just an ordered list. So:

- **"Chronological"** (above) -- by year.
- **"By architecture family"** -- group: linear methods
  (Perceptron, Adam), recurrence (RNN, LSTM, Mamba),
  convolution (LeNet, AlexNet, ResNet, U-Net), attention
  (Transformer, GPT, BERT, ViT, CLIP, MoE).
- **"By training paradigm"** -- group: supervised, unsupervised,
  self-supervised, RLHF, distillation. Maps onto the existing
  `HOW_MODELS_LEARN` lesson's spine.
- **"What MLPL can run today"** -- the runnable-only subset, for
  a learner who wants hands-on without conceptual detours.
- **"What MLPL cannot run yet (and why)"** -- the deferred-only
  subset, doubling as a roadmap.

Each path is one entry in `PATHS`; no new content needed.

## Relationship with the dim-reduction milestone (confirmed 2026-05-20)

`docs/milestone-dimensionality-reduction.md` ships three
focused per-method lessons -- "SNE: the very-slow ancestor",
"t-SNE: a peek at nonlinear methods", "UMAP: the modern default"
-- as part of its Phase 4. The chronological-history milestone
**references those three lessons by title** rather than
duplicating them. In the chronological path they slot in
between Word2Vec (2013) and Vision Transformer (2020) in
publication order:

- 2002 -- "SNE: the very-slow ancestor" (Hinton & Roweis)
- 2008 -- "t-SNE: a peek at nonlinear methods" (van der Maaten & Hinton)
- 2018 -- "UMAP: the modern default" (McInnes & Healy)

Implication: the dim-reduction milestone is a soft prerequisite
for this milestone's coverage of those three years. If this
milestone ships first, the three lesson slots in the path are
short stubs that get filled in when the dim-reduction
milestone's Phase 4 lands. If dim-reduction ships first, the
slots are populated from day one.

## Phases

### Phase 1: lesson scaffolding + the chronological path

Write the 24 lesson stubs (title, intro paragraph 1, "deferred"
or "MLPL example" placeholder). Wire them into `lessons.rs` or
`lessons_advanced.rs`. Add the `Chronological history` path. At
this stage the path is navigable end-to-end but most lessons
are short.

### Phase 2: flesh out the runnable lessons

The ~12 lessons MLPL can run get full intros + 3-5 examples +
try_it. Order: Perceptron, Backprop, Universal Approximation,
Word2Vec, Adam, ResNet, Transformer, GPT-1, ViT, then any
others where the runnable angle is clear.

### Phase 3: flesh out the deferred lessons

The ~12 deferred lessons get full intros explaining what the
paper introduced + a `Deferred:` paragraph naming the saga / RFC
needed to bring it home. These double as triage input for the
saga schedule.

### Phase 4: the alternative groupings

Add the four sibling paths (architecture family, training
paradigm, runnable-only, deferred-only). Each is ~50 lines in
`paths.rs`.

### Phase 5: glossary + diagram cross-references

Many of the 24 lessons already have matching glossary entries
and diagrams. Wire `[[term]]` sigils into the lesson intros so
the glossary-popup machinery works from the lesson view too;
add `Step::Diagram` interleavings to the chronological path
where the diagram set has the matching SVG.

## What I want to confirm before starting

- Whether **24 lessons** is roughly the right granularity, or
  whether you want it tighter (~12 lessons, fewer deferred
  entries) or wider (~40 lessons, more empirical papers like
  GPT-3 and PaLM).
- Whether the **deferred lessons** should ship in Phase 1
  alongside the runnable ones (so the path is complete on day
  one) or be added later (so the path grows incrementally).
- Whether the **alternative groupings** (Phase 4) are wanted
  now or whether the chronological path alone is enough to
  start.
