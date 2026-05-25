Saga 33 step 005: extract env_values.rs + env_tags.rs from env.rs.

Continue env.rs split (now ~22 methods after step 004).

Two related sibling files:

crates/mlpl-eval/src/env_values.rs (impl Environment block for per-value-type accessors):
- set_string, get_string
- set_record, get_record
- set_string_list, get_string_list
- set_builtin_ref, get_builtin_ref
(8 methods -- borderline FAIL since max is 7; if it FAILs, split further into env_values.rs + env_values_misc.rs)

crates/mlpl-eval/src/env_tags.rs (impl Environment block for tag operations):
- set_tag, get_tag, clear_tag, tags_iter
(4 methods -- PASS)

Register both in lib.rs.

Target: env.rs 22 -> 10 methods. Almost at the budget.

Strict gate: net-negative on BOTH fails AND warnings vs HEAD~1. If env_values ends up at 8 fns FAILing, split it within this step.