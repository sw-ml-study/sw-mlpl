# Saga: tech-debt spike -- retire the remaining 8 sw-checklist FAILs

Baseline 2026-07-26: 8 failed / 311 warnings. All eight are structural
(crate-module-count or module-fn-count boxed in by a crate cap), so
per docs/sw-checklist-paydown.md this is the dedicated-spike route:
one crate split per step, facade re-exports so consumers never change,
scripts/check-locks.sh --fix after every workspace membership change,
full gates per step, no behavior change.

Steps: 1) mlpl-viz (16 modules -> facade + viz-core + viz-marks +
viz-analysis); 2) mlpl-autograd (backward.rs 19 fns + reduction_ops.rs
11 fns, crate at module cap -> sibling ops crate + module splits);
3) mlpl-serve (17 modules); 4) mlpl-repl (13); 5) mlpl-web (10);
6) mlpl-web-eval (19, mind the #[path] test harnesses); 7) wrap-up:
counts, docs, and the plan for the one out-of-scope monster --
mlpl-eval at 100 modules, which is its own future saga (workspace
partition scale), documented not attempted here.