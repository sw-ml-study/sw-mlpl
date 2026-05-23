Saga 33 step 006: extract env_signals.rs from env.rs (final env.rs split).

Continue env.rs split (now ~10 methods after step 005).

Move to crates/mlpl-eval/src/env_signals.rs (impl Environment block for lifecycle signals):
- set_metric_sink, clear_metric_sink, metric_sink, emit_metrics
- set_interrupt, clear_interrupt, check_interrupt
(7 methods -- exactly at the budget, PASS)

Register in lib.rs.

After this step, env.rs should be down to:
- struct Environment { ... } definition + fields
- impl Environment { fn new() -> Self }
- the PeerDispatcher trait + its DefaultPeerDispatcher impl
(approximately 3 methods on Environment, plus the trait stuff -- well under 7).

Verify with: grep -c '^    pub fn \|^    fn \|^    pub(crate) fn ' crates/mlpl-eval/src/env.rs

If env.rs is at <=7 fns, the original Environment Module-Function-Count FAIL is RETIRED. That's the saga's main goal.

Strict gate: net-negative on BOTH fails AND warnings vs HEAD~1.