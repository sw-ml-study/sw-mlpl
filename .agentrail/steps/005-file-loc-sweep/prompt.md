Tech-debt saga step 005: File LOC sweep.

28 File-LOC FAILs at saga start (files over 500 lines). Many should have shrunk after steps 001-003; the rest need responsibility-bounded splits per docs/code_metrics.md.

For each remaining File-LOC FAIL:
- Diagnose the file's responsibilities. Identify the cleanest 2-way split (or 3-way).
- Use the file-naming convention: parse.rs, validate.rs, plan.rs, run.rs, render.rs.
- lib.rs and mod.rs stay facade-only (no executable logic).
- Move tests out of production modules when the inline cfg(test) block is hurting readability; prefer sibling tests/ directories or parse_tests.rs next to parse.rs.

Target retirement: -15 File-LOC FAILs.

Strict gate: sw-checklist net-negative on BOTH fails AND warnings vs HEAD~1.