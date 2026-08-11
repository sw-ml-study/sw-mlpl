# Saga: repl-help-and-demo-guidance

Two user-requested REPL/demo UX fixes that share one pages deploy
(both are web-visible: the web REPL runs the same eval `inspect()`,
and demos render in the browser).

1. `:help <name>` bug: `:help take` (any builtin) errors with "no
   help topic 'take'" because `help_topic` only matches category
   TOPICS (vars/models/fns/builtins/...), never builtins. Fix:
   `help_topic` falls back to `inspect_describe::format_describe`
   for a non-topic name, returning Some(describe) when the name
   resolves (builtin / var / model / fn / REPL command) and None
   (-> topic list) only for the "'x' is not ..." sentinel. No new
   function (inspect.rs is at its 7-fn max) -- extend help_topic.
   Add a completeness test: for EVERY documented builtin name,
   `:help <name>` returns real help (no "no help topic" / "is not
   a bound" sentinel), so :help stays complete as builtins land.

2. Demos "Next Steps?" epilogue: after a demo runs, show a short
   note suggesting `:help`, `:vars`, `:fns`, `:list`, and
   `experiment`. Render it ONCE in the demo runner (not per-demo in
   demos.toml).

## Steps
1. help-take-fix -- extend help_topic + completeness test; scoped
   mlpl-eval tests (inspect/help) green; clippy/fmt.
2. demo-next-steps -- append a "Next Steps?" epilogue in the demo
   render path; test; clippy/fmt.
3. deploy -- build-pages.sh, commit pages/, deploy-pages.sh, verify
   live + fresh Pages build if stalled, --done.
