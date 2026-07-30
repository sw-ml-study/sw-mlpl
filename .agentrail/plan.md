# Saga E1: engram-primitives (CPU reference + parity fixtures)

Per docs/engram-sagas-plan.md (decisions D1-D4). Goal: the hash /
gather primitive layer of Engram, CPU-first, with exact-parity
fixtures that the MLX backend (saga E5) must later reproduce
bit-for-bit.

Steps (draft):
1. design-note + component scaffold: components/engram/ with
   mlpl-engram-core (EngramSpec/HashSpec, validation, derived
   accounting -- no tensor deps) + hash reference fixtures.
2. array/runtime primitives: shift_pad, rolling n-gram
   mul-add-mod hash over f64 arrays (exact to 2^53), flattened
   multi-head gather with head-offset tables -- as general
   builtins (ngram_hash, gather_rows) in the runtime/eval path.
3. demo + docs: demos.toml "Engram" group, engram_hash demo
   (deterministic indices shown on CPU; the CPU==MLX acceptance
   gate is recorded as a fixture for E5), lang-reference/glossary
   entries.
