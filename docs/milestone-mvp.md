# MVP

The MLPL MVP is a Rust-first tensor/array language with:

- [x] dense arrays
- [x] basic tensor operations (reshape, transpose)
- [x] broadcasting (scalar-to-array)
- [x] reductions (reduce_add, reduce_mul, axis-specific)
- [ ] rank/cell semantics (deferred to post-MVP)
- [x] trace export (JSON via :trace json)
- [ ] a browser-based visual trace viewer (deferred to post-MVP)

## What ships in MVP

- Interactive REPL with :help, :trace, :clear
- Script file execution (-f flag)
- 9 built-in functions
- Element-wise arithmetic with scalar broadcasting
- Unary negation
- Execution tracing with JSON export
- Demo scripts in demos/
