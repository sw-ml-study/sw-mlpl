Final MVP validation: run all demos, verify traces, confirm docs.

1. Run each demo script via the REPL --file flag and verify output
2. Run a demo with :trace on and export JSON, verify it's valid
3. Run cargo test across the entire workspace -- all must pass
4. Run all quality gates (clippy, fmt, markdown-checker, sw-checklist)
5. Verify README accurately describes current capabilities
6. Tag the repo as v0.1.0-mvp

This is the acceptance test for the MVP milestone.

Fix any bugs found during validation.

Allowed: all directories
