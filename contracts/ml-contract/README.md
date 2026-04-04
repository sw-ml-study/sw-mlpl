# ML Contract

## Purpose

Define the behavioral spec for machine learning primitives in MLPL.
`mlpl-ml` provides ML-specific operations built on top of the array
and runtime layers. This is the outermost layer in the dependency
chain and is later-phase scope.

## Key Types and Concepts

### Planned Capabilities

- Gradient computation (autograd or manual)
- Loss functions
- Optimizers (SGD, Adam, etc.)
- Layer abstractions (dense, activation, etc.)
- Training loops

## Invariants

- ML operations compose from array and runtime primitives
- Numerics must be deterministic for reproducibility

## What This Contract Does NOT Cover

- Array storage (that is `mlpl-array`)
- Expression evaluation (that is `mlpl-eval`)
- Parsing
- Visualization

## Open Questions

- Whether autograd is trace-based or tape-based
- Scope of first ML milestone (single-layer net? logistic regression?)
- Whether ML ops are built-in functions or a separate dispatch layer
