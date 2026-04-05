# MLPL Language Core Saga

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

Build the MLPL language pipeline from lexer to working REPL. At the
end of this saga, the REPL should evaluate the full syntax-core-v1.md
example set: array literals, arithmetic, variables, function calls
(reshape, transpose, shape, iota, reduce_add).

## What already exists (from mlpl-foundation saga)

- mlpl-core: Span, Identifier
- mlpl-array: Shape, DenseArray, reshape, transpose, indexing, Display
- mlpl-parser: lexer with all v1 tokens
- mlpl-eval: PoC evaluator (number sequences only)
- mlpl-repl: PoC REPL (lex -> eval -> print)

## Phases

### Phase 1: Parser AST and expression parsing
- Define AST node types
- Parse literals and array literals
- Parse arithmetic with precedence
- Parse function calls and assignment

### Phase 2: Evaluator and runtime
- Real evaluator that walks the AST
- Environment for variable bindings
- Element-wise arithmetic with scalar broadcasting
- Built-in function registry (reshape, transpose, shape, iota, reduce_add, reduce_mul, rank)

### Phase 3: REPL integration
- Wire parser + evaluator into REPL
- Multi-statement support
- Error display with source spans

## Success criteria
- All syntax-core-v1.md examples work in the REPL
- cargo test passes with real tests for parser, eval, runtime
- cargo run -p mlpl-repl is a usable interactive calculator
