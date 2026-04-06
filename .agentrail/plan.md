# MLPL MatMul and ML Foundations Saga

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

Add matrix multiplication and math functions to MLPL, then
demonstrate training a logistic regression model on a toy dataset.
This is the first step toward ML in the language.

At the end of this saga, the REPL should be able to:
- Multiply matrices: matmul(A, B)
- Apply math functions: exp, log, sigmoid, sqrt, abs
- Run a manual gradient descent training loop
- Train a logistic regression model and show accuracy

## What already exists

- mlpl-array: DenseArray with reshape, transpose, element-wise ops,
  scalar broadcasting, axis reductions
- mlpl-runtime: 9 built-in functions
- mlpl-eval: AST-walking evaluator with environment
- mlpl-parser: full v1 syntax (literals, arrays, arithmetic,
  function calls, assignment, unary negation)
- mlpl-repl: REPL with :help, :trace, :clear, -f flag

## Phases

### Phase 1: Linear algebra
- Dot product for vectors
- Matrix multiplication
- Element-wise math functions

### Phase 2: ML primitives
- Sigmoid activation
- Binary cross-entropy loss
- Manual gradient computation

### Phase 3: Training demo
- Logistic regression training loop
- Toy dataset (AND/OR gate or simple 2D classification)
- Accuracy measurement

## Success criteria
- matmul(A, B) works for compatible matrices
- Math functions (exp, log, sqrt, abs, sigmoid) work element-wise
- A training script in demos/ converges on a toy problem
- All tests pass
