# Saga: data-forge (Track 1, per roadmap revision B)

The learning-experience-infrastructure saga from the 20260802
direction research: synthetic data as a first-class ML algorithm.
Generators -> evaluators -> rejection/ranking -> curation/
curriculum -> provenance, plus the MINIMAL knowledge-graph task
infrastructure. Everything array-first: examples are token-id
sequences and numeric records, NOT text (the strings gap G3 stays
out of scope; teacher-text generation waits for it).

Design constraints:
- Array-first examples (id sequences / numeric rows) so the whole
  pipeline works inside MLPL's existing value system (Array,
  Record, Result, u: functions, :builtin refs, experiment blocks).
- Evaluators are ordinary functions (u: or builtin refs) returning
  scores; deterministic oracles first (exact match, graph-path
  verification, arithmetic checkers); LLM-judge waits for the
  agent track.
- Reuse before invent: val_split, shift_pairs_*, sample, top_k,
  experiment tracking, one_hot, knn already exist.
- The rejection/ranking substrate is the cross-cutting primitive
  later tracks reuse (MTP proposal analysis, ICRL, distillation).

## Steps

1. design-surface -- docs/data-forge-design.md: the MLPL surface
   (names, arities, semantics, worked examples), what is
   composable today vs new builtins, the knowledge-graph value
   representation, provenance record shape. USER REVIEW GATE:
   present the surface before implementing.
2..N (refined after review): kg-core; generators;
   evaluators + rejection substrate; curation + curriculum;
   provenance + demos (arithmetic curriculum, graph multi-hop);
   close.
