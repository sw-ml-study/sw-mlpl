Phase 3 step 008: REPL introspection for experiments. Add :experiments and compare() commands.

:experiments command (terminal + web): lists every recorded run in order with its name, timestamp, top-line metric (the first _metric-suffixed scalar alphabetically, or "(no metrics)"). In the terminal REPL, lists runs from both the in-environment log AND the on-disk exp_dir (merge by timestamp). In the web REPL, just the in-environment log.

compare(name_a, name_b) builtin (NOT a : command -- it returns a Value::Str so it can be called from MLPL code too): prints a side-by-side of the latest run with each name, showing each metric as "metric_name: a_value vs b_value (delta)". Runs named identically with multiple timestamps -- use the most recent.

A new inspect fn format_experiments(env) in mlpl-eval/src/inspect.rs produces the :experiments output. compare() lives in eval.rs or eval_ops.rs as a regular builtin.

Plumbing: needs a reader for .mlpl-exp/*/run.json when running the terminal REPL. Reuse serde_json (already a workspace dep via mlpl-trace). Handle a missing or malformed run.json by skipping it with a warning.

TDD:
(1) :experiments with no runs returns "(no experiments recorded)".
(2) after two experiment blocks named "a" and "b", :experiments lists both in order.
(3) compare("a", "b") returns a Value::Str mentioning both names and their respective metrics.
(4) compare on a missing name errors clearly.
(5) terminal REPL path: write a stub run.json by hand into a temp exp_dir, :experiments picks it up.
(6) web REPL path: :experiments sources only from env.experiment_log.
