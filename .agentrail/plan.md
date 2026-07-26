# Future saga: Classical ML in an array language (kNN + Naive Bayes)

Seed input for `agentrail init`. Captures (1) how the playground's
demos, tutorials, and glossary map onto a classical-ML book
curriculum, (2) the gaps that mapping exposes, and (3) a concrete plan
for the two gaps worth closing with hands-on demos. Reference
curriculum: Akimenko, "Machine Learning From Scratch" (Linear
Regression, Logistic Regression, Regularization, K-Nearest Neighbors,
Naive Bayes, Tree Algorithms, Decision Tree, Ensembles, Random Forest,
Gradient Boosting, XGBoost, Neural Network).

## Coverage analysis (2026-07-25)

The book and the playground share a spine -- linear models ->
regularization -> neural networks -- but walk opposite branches from
it. The book goes deep on non-differentiable classical algorithms;
MLPL is a differentiable-programming platform, so everything it
teaches runs through gradient descent, and the classical branch is
covered only at glossary depth.

| Book chapter | Playground coverage | Where |
| --- | --- | --- |
| Linear Regression | Implicit, via GD | "How Gradient Descent Works" demo fits y = w*x + b on a loss surface; "What is ML" / "Why backprop?" lessons. No closed-form / normal-equation treatment. |
| Logistic Regression | Strong | "Logistic Regression" demo, the lesson of the same name, both "Decision Boundary" demos. |
| Regularization | Strong | "Taming Overfitting: Weight Decay" demo, the Watch a Model Learn / Generalize pair, glossary Regularization + Weight Decay. L2 flavor only (no ridge/lasso naming, no L1). |
| K-Nearest Neighbors | Glossary-adjacent only | Neighbor-graph ideas in the embedding lesson and UMAP / t-SNE demos; no kNN classifier anywhere. |
| Naive Bayes | Absent | One passing glossary mention. |
| Decision Tree | Glossary entry only | Real entry (splits, information gain, Gini) -- context, no construct. |
| Random Forest / Ensembles | Glossary entry only | Bagging / voting described; nothing hands-on. |
| Gradient Boosting / XGBoost | Mentions only | Referenced inside ensemble entries; no dedicated entry. |
| Neural Network | Platform core, far past the book | MLP / CNN / RNN / LSTM / GAN / autoencoder / transformer ladder / ViT / diffusion / LoRA demos, autograd + optimizer lessons, the beginner spine path, live loss telemetry. |

A reader working through the book can use the playground as a lab for
the linear/logistic/regularization chapters and the neural-network
chapter. The middle chapters (kNN through XGBoost) have no hands-on
counterpart -- mostly a deliberate gap, since those algorithms do not
exercise the autograd tape. But two of them are natural, cheap wins
in an array language, and closing them makes the book-parallel story
honest: "classical algorithms ARE array programs too."

## Glossary gaps (verified against H2 entries)

Missing outright: **Linear Regression, Logistic Regression, k-Nearest
Neighbors, Naive Bayes, Ensemble Learning, Gradient Boosting,
XGBoost.** Present already: Decision Tree, Random Forest,
Regularization, Weight Decay, MLP, Perceptron, CNN, RNN. Each new
entry follows house style: what it is, the closest MLPL construct (or
"deferred" marker), cross-links via [[term]].

Note the count ratchet: `readme_counts` pins README's "367-entry
glossary" to the H2 count in `docs/glossary.md`; adding 7 entries
means updating both README mentions in the same commit.

## Why kNN and Naive Bayes (and not trees/boosting)

- **kNN is a distance matrix.** `d2 = |a|^2 + |b|^2 - 2ab` is one
  `matmul` plus two `reduce_add`s; neighbor selection is `top_k` of
  the negated distances; voting is `one_hot` + `reduce_add` + `argmax`.
  Every builtin already exists. Deeply APL-spirited: the whole
  classifier is ~6 lines with no loops.
- **Gaussian Naive Bayes is masked means.** Class masks via
  `eq(Y, c)`; per-class feature means/variances via mask matmuls;
  prediction is `argmax` over summed log-densities (`log`, `exp`,
  `sqrt` all exist). Again pure array ops, no new builtins.
- **Trees / forests / boosting fit the language poorly**: greedy split
  search is control-flow-heavy and non-differentiable. The tic-tac-toe
  minimax proves MLPL *can* express such recursion, but a faithful
  CART/GBM demo would be long, slow, and teach control flow rather
  than array thinking. They stay glossary-level, with the new
  Ensemble/Boosting entries pointing at the Random Forest entry for
  intuition.

## Saga steps

1. **glossary-classical-terms** -- add the 7 missing entries (house
   style, [[cross-links]] to the new demos by name), update both
   README glossary counts, `markdown-checker`, and the
   `readme_counts` test stays green. No wasm rebuild yet (batched
   with step 4's pages rebuild).
2. **knn-demo** -- "K-Nearest Neighbors" demo in a new "Classical ML"
   demo category: `blobs` dataset, distance-matrix construction shown
   step by step, `top_k` neighbor pick, `one_hot` vote, accuracy vs a
   held-out split, `scatter` of predictions vs truth. TDD: an eval
   test asserting kNN accuracy beats a majority-class baseline on
   blobs; a demos-smoke registry entry. Lesson tie-in: short
   "Classical ML: distances instead of gradients" framing note in the
   demo intro; glossary [[k-Nearest Neighbors]] link in a line comment.
3. **naive-bayes-demo** -- "Naive Bayes (Gaussian)" demo, same
   category: class masks, per-class mean/variance via masked matmuls,
   log-density scoring, `argmax` posterior, decision-boundary render
   vs the logistic-regression demo on the same moons data (the
   pedagogical payoff: generative vs discriminative on one dataset).
   TDD as step 2.
4. **paths-docs-pages** -- register both demos in the taxonomy +
   a "Classical ML detour" stop on the beginner spine path,
   lang-reference note, pages/ rebuild, CHANGES refresh, retrospective.

Per-step budget discipline: each demo lands as demos.toml content plus
one eval test file; no new crates; sw-checklist ratchet paired per
commit as usual.

## Risks / notes

- `top_k` semantics on ties should be pinned by the eval test.
- Distance matmul on [200, 2] blobs is trivial CPU work -- both demos
  stay browser-tier (no connect requirement), which also means they
  join the public live demo on merge.
- Keep the "Classical ML" category small and pointed; tree/boosting
  demos are explicitly out of scope (revisit only if a saga wants a
  control-flow showcase).
