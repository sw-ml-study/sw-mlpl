# MLPL MVP Completion Saga

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
4. Use /mw-cp for checkpoint process (checks, detailed commit, push)
5. Push immediately after commit

## Goal

Complete the MLPL MVP: add the remaining language features
(rank/cell semantics, axis reductions, unary negation), build the
trace system with JSON export, and create compelling CLI demos.

At the end of this saga, MLPL should be demonstrable as a working
array language with execution tracing.

## What already exists (from previous sagas)

- mlpl-core: Span, Identifier
- mlpl-array: Shape, DenseArray, reshape, transpose, indexing,
  element-wise ops with scalar broadcasting, Display
- mlpl-parser: lexer (all v1 tokens), parser (AST with literals,
  arrays, arithmetic with precedence, function calls, assignment)
- mlpl-eval: AST-walking evaluator with environment
- mlpl-runtime: builtins (iota, shape, rank, reshape, transpose,
  reduce_add, reduce_mul)
- mlpl-repl: working REPL with full syntax-core-v1 support

## Phases

### Phase 1: Language enrichment
- Unary negation in expressions
- Axis-specific reductions (reduce along an axis)
- Element-wise comparison operators (future consideration)

### Phase 2: Trace system
- TraceEvent and Trace types in mlpl-trace
- Instrument evaluator to emit trace events
- JSON serialization of traces (serde)
- Trace export command in REPL

### Phase 3: CLI demos and polish
- Demo scripts exercising all features
- REPL help command
- Error messages with source context
- Update documentation

## Success criteria
- All v1 syntax examples work
- Trace JSON output for any evaluation
- cargo test passes for all crates including trace
- Demo scripts run and produce correct output
