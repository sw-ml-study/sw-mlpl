Step 002: Eval: function storage + call dispatch.

Store defined functions in Environment (new field: user_fns HashMap). On call to a colon-containing name, look up the FnDef, create a child scope with args bound, evaluate the body, return the last value. Handle Return as a control-flow signal (like Break). TDD: write eval tests (simple call, recursion, lexical scoping, arity mismatch).