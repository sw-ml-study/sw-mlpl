Saga 33 step 002: extract env_vars.rs from env.rs.

crates/mlpl-eval/src/env.rs has 55 methods on Environment (Module-Function-Count FAIL by ~48). Per docs/loose-coupling.md, Rust allows multiple impl Environment blocks across modules in the same crate. Each block can host a topical group of methods.

This step: extract the var + param cluster (10 methods).

Move to crates/mlpl-eval/src/env_vars.rs (new file with 'impl Environment { ... }' block):
- get(name), set(name, value), set_param(name, value)
- mark_param, is_param, mark_frozen, unmark_frozen, is_frozen
- params (iter), vars_iter

Register 'mod env_vars;' in lib.rs. The Environment struct + fields stay in env.rs. The methods accessing pub(crate) fields work because they're in the same crate.

Target: env.rs 55 -> 45 methods. Still FAIL (>7) but progress. env_vars.rs: 10 fns -> still FAIL (>7) -- may need to split into env_vars.rs + env_params.rs (5 each) within this step OR a follow-up step. Try the simple version first; check sw-checklist; split further only if env_vars.rs FAILs alone.

Strict gate: net-negative on BOTH fails AND warnings vs HEAD~1.