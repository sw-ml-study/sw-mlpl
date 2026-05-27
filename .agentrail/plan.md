# User-Defined Functions + Control Flow

Proposed milestone (saga 46).

## What exists today

MLPL has these control-flow constructs (all added by saga 31):

- `if cond { then } else { else_ }` -- conditional expression
- `while cond { body }` -- conditional loop
- `break` -- exit a while loop early
- `repeat N { body }` -- counted loop
- `train N { body }` -- counted loop with loss tracking + adam
- `for x in expr { body }` -- iteration over arrays

MLPL has NO user-defined functions. All callable names are
builtins hardcoded in Rust (`match name { ... }` in
`builtins.rs` and `eval_fncalls.rs`).

## What this saga adds

### 1. User-defined functions (`def`)

```
def u:circle_area(r) {
    pi() * r * r
}
u:circle_area(5)
```

Key design decisions:

- **Namespace**: colon separator. `u:name` for end users,
  `vendor:name` for add-on packages. Names without `:` are
  reserved for builtins. Parser enforces this: `def area(r)`
  is rejected because `area` has no namespace prefix.

- **Syntax**: `def prefix:name(arg1, arg2, ...) { body }`.
  Body is a sequence of expressions; the last expression's
  value is the return value.

- **Scoping**: lexical. Parameters are local to the function
  body. The function can read (but not write) variables from
  the enclosing scope (closure over the environment at
  definition time).

- **Recursion**: allowed. The function name is bound before
  the body executes, so `def u:fib(n) { if gt(n, 1) { ... }
  else { n } }` works.

- **No overloading**: one definition per name. Re-defining
  overwrites silently (like variable assignment).

- **No variadic args**: fixed arity, checked at call site.

### 2. `return` (optional, for early exit)

```
def u:safe_div(a, b) {
    if eq(b, 0) { return 0 }
    a / b
}
```

Without `return`, the last expression is the return value
(like Rust). `return expr` exits early with `expr`.

### 3. `pi()` and `e()` zero-arg builtins

Trivial additions. `pi()` returns 3.14159265358979...,
`e()` returns 2.71828182845904.... Both are reserved
builtin names (no `:` prefix).

### 4. `match` / pattern dispatch (stretch goal)

```
match shape(x) {
    [1] => "scalar"
    [n] => "vector of " + str(n)
    [r, c] => "matrix"
    _ => "higher rank"
}
```

This is a stretch goal. If it complicates the parser
significantly, defer to a later saga. The `if/else` chain
covers all cases, just less elegantly.

## What this saga does NOT add

- **Closures as values** (passing functions as arguments).
  Requires a `Value::Function` variant and higher-order
  dispatch. Deferred.
- **`do-while`**. `while` + initial setup covers all cases.
- **`select`/`case`**. `match` is the modern version.
- **Type annotations**. MLPL is dynamically typed.
- **Modules / imports**. The `vendor:` prefix is the
  namespace mechanism; actual module loading is deferred.

## Suggested steps

1. **Parser: `def` + `return`**. Add `Expr::FnDef` and
   `Expr::Return` AST nodes. Parse `def prefix:name(args)
   { body }`. Reject names without `:` prefix. TDD.

2. **Eval: function storage + call dispatch**. Store
   defined functions in `Environment`. On call, create a
   child scope with args bound, evaluate the body, return
   the last value. Handle `return` as a control-flow
   signal (like `break`). TDD.

3. **`pi()` and `e()` builtins**. Trivial zero-arg math
   constants. Add to math_builtins.rs NAMES, try_call,
   inspect_groups, lang-reference, help.rs. TDD.

4. **Recursion + scoping tests**. Fibonacci, factorial,
   mutual recursion (if two `def`s reference each other).
   Verify lexical scoping: inner function reads outer vars
   but does not mutate them.

5. **Demo + glossary + path entry + saga close**. A demo
   showing UDF definition and use. Glossary entries for
   Function, Scope, Recursion. Update help text.

## Quality requirements

Same as all sagas. TDD. sw-checklist budgets. Every commit
reduces warnings.

## Dependencies

None. All prerequisite control flow (`if/else`, `while`,
`for`, `break`) already exists.
