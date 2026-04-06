Add trace export to the REPL.

1. Add a :trace command to the REPL:
   - ":trace on" enables tracing for subsequent evaluations
   - ":trace off" disables tracing
   - ":trace" shows the last trace as formatted text
   - ":trace json" exports the last trace as JSON to stdout
   - ":trace json <filename>" writes JSON to a file

2. When tracing is on:
   - Each line evaluation captures a Trace via eval_program_traced
   - The trace is stored and available for :trace commands

3. Default: tracing is off (no overhead when not needed)

TDD:
- Integration test: eval with :trace on, verify trace is captured
- Test :trace json output is valid JSON
- Test :trace off stops capturing
- Test that regular eval (no tracing) still works

Allowed: apps/mlpl-repl/, crates/mlpl-eval/
May read: crates/mlpl-trace/
