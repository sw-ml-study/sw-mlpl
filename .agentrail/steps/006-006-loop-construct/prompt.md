Add a simple loop construct to support training iterations.

The REPL currently has no looping. For a training demo, we need
a way to repeat operations. Add a minimal "repeat" built-in
or a for-loop construct.

Option A (simpler): Add a "repeat" built-in function:
  repeat(n, "statement1; statement2; ...")
  - Takes an integer count and a string of statements
  - Evaluates the string n times in the current environment
  - Returns the last result

Option B (better): Add Expr::ForLoop to the AST:
  for i in iota(100) { body }
  - But this requires new syntax, lexer tokens, parser changes

Go with Option A for now -- it's sufficient for a training demo
and doesn't require parser changes. The string is parsed and
evaluated each iteration.

Actually, simplest approach: just run the training script with
many repeated lines via -f flag. Skip the loop construct for now
and handle iteration in the demo script by writing explicit lines
or by adding a "train" higher-level built-in.

Revised approach: Add a "train_step" concept as a multi-line
eval. Add eval_n(n, source_string) built-in that parses and
evaluates a source string n times. Variables persist between
iterations.

TDD:
- eval_n(3, "x = x + 1") with x=0 -> x is 3
- eval_n(10, "w = w - 0.1") with w=1 -> w is ~0.0
- eval_n(0, "x = 1") -> no-op

Allowed: crates/mlpl-runtime/, crates/mlpl-eval/, crates/mlpl-parser/
