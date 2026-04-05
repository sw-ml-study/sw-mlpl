Build a REPL proof-of-concept that demonstrates end-to-end evaluation of simple array literals.

Goal: type "2 3 4" into a prompt and see it parsed, evaluated, and printed as a 1-D array.

Implementation:
1. In crates/mlpl-eval/src/lib.rs, add a minimal evaluate() function:
   - Takes a Vec<Token> from the lexer
   - Recognizes a sequence of numeric literals as an array literal
   - Returns a DenseArray (from mlpl-array)
   - Errors on anything it doesn't understand (that's fine for PoC)

2. In apps/mlpl-repl/src/main.rs, wire up a read-eval-print loop:
   - Use std::io for line reading (no rustyline yet)
   - Lex the input line (mlpl-parser lexer)
   - Evaluate (mlpl-eval)
   - Print the result (Debug or a simple Display impl on DenseArray)
   - Loop until EOF or "exit"

3. Add Display impl for DenseArray if not already present:
   - 1-D: "2 3 4"
   - 2-D: show rows on separate lines (stretch goal, not required)

TDD:
- Test evaluate() with "1 2 3" tokens -> DenseArray with shape [3]
- Test evaluate() with single number -> DenseArray with shape [1] or scalar
- Test evaluate() with empty input -> appropriate error
- Test evaluate() with unknown tokens -> error

This is a PoC -- keep it minimal. No fancy error recovery, no editing,
no history. Just prove the pipeline works end-to-end.

Allowed directories: crates/mlpl-eval/, apps/mlpl-repl/
May read (not modify): crates/mlpl-parser/, crates/mlpl-array/, crates/mlpl-core/

Verify:
- cargo test passes for all crates
- cargo run -p mlpl-repl lets you type "1 2 3" and see output
- cargo clippy --all-targets --all-features -- -D warnings