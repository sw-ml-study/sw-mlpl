Phase 3 step 007: experiment "name" { body } scoped form + metric capture.

Parser: add Token::Experiment keyword; add Expr::Experiment { name: String, body: Vec<Expr>, span: Span }. Grammar: `experiment STRING_LIT { body }`.

Evaluator: eval_experiment runs the body in a child scope. On entry, snapshot: (a) current PRNG seed state if the environment has a notion of one (Saga 10's random/randn builtins are seed-arg based; document that experiment captures "seed reproducibility" by recording all explicit seed args used inside the block -- simplest viable design), (b) any currently-bound params' shapes + labeled shapes, (c) the body's source span so we can later extract the source text. On exit, scan the environment for any scalar variable whose name ends in _metric and record them as the run's metrics.

Recording: terminal REPL writes a JSON file at <exp_dir>/<name>/<unix-nanos>/run.json where exp_dir is the new --exp-dir CLI flag (default ./.mlpl-exp). Web REPL appends a record to an in-environment Vec<ExperimentRecord> exposed as env.experiment_log.

JSON shape: { "name": "...", "timestamp_ns": N, "source": "...", "metrics": { "loss_metric": 0.123, ... }, "params_snapshot": { "W": { "shape": [2, 3], "labels": [null, "feat"] }, ... } }.

Compile path: Expr::Experiment returns LowerError::Unsupported (experiments are a REPL-time reproducibility construct, not a compiled-code one).

TDD:
(1) parse `experiment "test" { x = 1 }` into Expr::Experiment.
(2) running a minimal experiment in the terminal REPL writes a valid run.json; its metrics dict includes any _metric-suffixed scalars.
(3) experiment with no body errors clearly.
(4) experiment with no _metric-suffixed names writes an empty metrics dict, not a missing key.
(5) web REPL path: env.experiment_log gains one entry with the same shape.
(6) lower-rs rejects Expr::Experiment as Unsupported.
