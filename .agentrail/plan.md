# MLPL ML Demos Saga

## Quality Requirements (apply to EVERY step)

Every step MUST:
1. Follow TDD: write failing tests FIRST, then implement, then refactor
2. Pass all quality gates before committing:
   - cargo test (ALL tests pass)
   - cargo clippy --all-targets --all-features -- -D warnings (ZERO warnings)
   - cargo fmt --all (formatted)
   - markdown-checker -f "**/*.md" (if docs changed)
   - sw-checklist (project standards)
3. Update relevant docs if behavior changed
4. Use /mw-cp for checkpoint process
5. Push immediately after commit
6. Steps that touch the web UI must rebuild pages/ via scripts/build-pages.sh

## Goal

Stand up a small library of classic ML demos that exercise the
v0.3 visualization stack end-to-end. Users land on the browser
REPL, pick a demo from the dropdown, and see a familiar algorithm
(k-means, PCA, softmax, a tiny MLP, an attention pattern) train
and produce a publication-quality SVG inline.

This saga is the payoff for the SVG visualization saga: the
diagrams already work, but MLPL is missing the synthetic-data
primitives (random numbers, gaussian blobs, etc.) and the
algorithmic glue needed to write the demos cleanly.

## What already exists

- mlpl-array, mlpl-runtime, mlpl-eval, mlpl-parser, mlpl-trace
- mlpl-viz with svg() + 5 high-level analysis helpers
  (hist, scatter_labeled, loss_curve, confusion_matrix,
  boundary_2d) and grid()
- Browser REPL with inline SVG rendering and SVG download button
- Tutorial with 12 lessons; demo dropdown with 8 entries
- Logistic regression and decision-boundary demos

## Phases

### Phase 1: Synthetic data primitives
- random(seed, shape): seeded uniform random array
- randn(seed, shape): seeded standard-normal random array
- blobs(seed, n_per_class, centers): canned 2D gaussian-blob
  dataset returning Nx2 points and a length-N label vector
- argmax(a, axis): index of max along an axis (needed by
  classifiers and k-means assignment)

### Phase 2: Algorithm demos
- k-means clustering on a blobs dataset, rendered with
  scatter_labeled and the cluster centers overlaid
- PCA on a small synthetic dataset, rendered as a 2D scatter
- Softmax + cross-entropy classifier on a 3-class blobs dataset,
  with a confusion matrix and a per-class decision-boundary view
- A tiny 1-hidden-layer MLP for the same 3-class problem, showing
  that the boundary becomes non-linear

### Phase 3: Attention pattern
- A toy "attention" demo: build a query/key dot-product matrix,
  softmax along rows, render the resulting attention pattern as
  a heatmap. No real model -- just the pattern.

### Phase 4: Polish + release
- Update the tutorial with 1-2 lessons that walk through the new
  demos at a beginner level
- Update milestone doc, README, and the demo dropdown
- Tag v0.4.0-demos

## Success criteria

- random/randn/argmax/blobs all available as built-ins, with tests
- k-means demo converges and renders cleanly inline
- PCA demo shows clear separation along the principal axis
- 3-class softmax classifier reaches > 95 percent accuracy on a
  separable blobs dataset
- Tiny MLP outperforms the linear classifier on a non-linearly
  separable dataset
- Attention pattern renders as a recognizable heatmap
- All tests pass, all quality gates green, pages deployed
- Tutorial includes at least one lesson per phase 2 demo
