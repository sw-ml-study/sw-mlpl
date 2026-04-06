Instrument the evaluator to emit trace events.

1. Add a Tracer to the evaluation context:
   - eval_program and eval_expr accept an optional &mut Trace
   - Each eval_expr call records a TraceEvent with:
     - The operation type (literal, binop, fncall, assign, ident)
     - The source span
     - Input values (for binop: lhs and rhs; for fncall: args)
     - Output value (the result)
   - Sequence numbers auto-increment

2. Update the public API:
   - eval_program(stmts, env) still works (no trace, backward compatible)
   - eval_program_traced(stmts, env, trace) captures the trace

3. The REPL does NOT use tracing yet (that comes in the next step).
   This step just makes it possible.

TDD:
- Eval "1 + 2" with tracing -> Trace has events for: IntLit(1), IntLit(2), Add
- Eval "x = [1,2,3]" -> Trace has events for array literal and assign
- Eval "iota(5)" -> Trace has events for IntLit(5) and FnCall("iota")
- Verify spans in trace events match source positions
- eval_program without trace still works (no regression)

Allowed: crates/mlpl-eval/, crates/mlpl-trace/
