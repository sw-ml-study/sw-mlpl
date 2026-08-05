# Saga: experiment-quality
Track 1 item 3 (docs/future-sagas-queue.md): evaluation rigor
before MTP/small-model conclusions. Two pillars: robustness
suites (array-shaped distribution-shift / format perturbations
as in-language idioms) and Pareto frontier analysis (native
pareto_front over experiment-block metrics). Full design:
docs/experiment-quality-design.md.
## Steps
1. design -- docs/experiment-quality-design.md; pause for user
   review of the three builtins + open questions.
2. pareto-core -- pareto_front(P, dirs) in mlpl-runtime-array, TDD.
3. experiment-bridge -- param_count(m) + experiment_metric("name")
   at the eval layer, TDD.
4. demos -- "Experiment Quality" category: Robustness Suite,
   Scaffold Dependence, Pareto Frontier; visual verification of
   every rendered SVG; pages deploy.
5. close -- lang-reference, glossary, queue advance, wiki errata.
