Wire the full parser + evaluator pipeline into the REPL, replacing the PoC.

Update apps/mlpl-repl/src/main.rs:
1. Use mlpl_parser::parse() instead of just lex()
2. Use the real mlpl_eval evaluator with an Environment that persists across lines
3. Variables assigned on one line are available on the next
4. Print the result of the last expression in each line
5. Show errors with the source span context when possible
6. Print a welcome message with version and available commands

The REPL should now handle the full syntax-core-v1.md example set:
- Scalar arithmetic: 1 + 2
- Vector arithmetic: [1, 2, 3] + [4, 5, 6]
- Scalar broadcast: [1, 2, 3] * 10
- Variables: x = iota(12); m = reshape(x, [3, 4])
- Function calls: transpose(m), shape(t), reduce_add([1,2,3,4,5])
- Multi-step: data = [1,2,3,4,5,6]; grid = reshape(data, [2,3]); scaled = grid * 2; result = reduce_add(scaled)

TDD: test the full pipeline with end-to-end integration tests in apps/mlpl-repl/tests/ or crates/mlpl-eval/tests/ that parse+eval complete programs.

Allowed: apps/mlpl-repl/, crates/mlpl-eval/
May read: all crates
