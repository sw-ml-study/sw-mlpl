# Appendix: Rust ML Crates

MLPL is built in Rust and draws inspiration from the burgeoning Rust ML ecosystem.

## Key Influences

- **Burn:** The focus on backend-agnostic tensor operations.
- **Candle:** The minimalist approach to model definitions.
- **ndarray:** The rigorous approach to multidimensional arrays.

## Why not just use Rust?

While Rust is excellent for production-grade machine learning systems, it can be verbose for rapid experimentation. MLPL acts as a "high-level DSL" for the logic you would eventually implement in Rust. 

MLPL handles the "what" (the transformations), while the underlying Rust implementation handles the "how" (the memory safety and performance).
