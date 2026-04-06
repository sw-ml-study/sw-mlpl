Create demo scripts and update documentation for MVP.

1. Create demos/ directory with MLPL script files:
   - demos/basics.mlpl: arithmetic, arrays, variables
   - demos/matrix_ops.mlpl: iota, reshape, transpose, shape
   - demos/computation.mlpl: the multi-step "42" computation
   - demos/trace_demo.mlpl: a computation designed to show tracing

2. Add a --file/-f flag to mlpl-repl to execute a script file:
   - cargo run -p mlpl-repl -- -f demos/basics.mlpl
   - Print each line and its result
   - Exit after file is complete

3. Update docs:
   - Update README.md with current capabilities, build instructions,
     and demo commands
   - Update docs/milestone-mvp.md to mark completed items
   - Update docs/saga.md to reflect completed sagas

TDD:
- Test --file flag reads and evaluates a script
- Test script output matches expected results
- Markdown validation on all updated docs

Allowed: all directories
