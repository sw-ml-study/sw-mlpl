Step 002: Shrink over-50-LOC functions.

Target: math_builtins.rs (8 fns -> split), ast_fmt fmt() (62 lines -> extract helpers), stmts.rs (7 fns -> split parse_def). Run sw-checklist before and after.