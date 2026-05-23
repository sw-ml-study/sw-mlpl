Tech-debt saga step 004: Function LOC sweep.

With steps 001-003 done, the new crate structure provides room to extract helpers. Walk the remaining Function-LOC FAILs (32 at saga start, expect ~25 here) and extract responsibility-bounded helpers into sibling modules.

Approach per docs/code_metrics.md:
- Each fn over 50 LOC = a FAIL. Diagnose what 2-3 jobs it is doing; extract the right one into a named sibling module.
- File-naming convention: parse.rs (input -> typed), validate.rs (typed -> result), plan.rs (config -> plan), run.rs (effects), render.rs (data -> string), error.rs, model.rs, test_support.rs, fixtures.rs.
- Do NOT compress via single-line struct literals -- rustfmt will revert them. Use field-init shorthand + tuple destructuring + closures + helper fns.

Target retirement: -15 Function-LOC FAILs, -20 Function-LOC warnings.

Strict gate: sw-checklist net-negative on BOTH fails AND warnings vs HEAD~1. If the step can't beat the prior commit, the scope is wrong -- narrow it.