Saga 21.5 step 003: cancellation.

Goal: ship POST /v1/sessions/<id>/cancel that flips a session-scoped AtomicBool. Thread an Interrupt token through mlpl_eval::eval_program so it checks the bool at the head of every loop iteration (for, train, repeat) plus before every builtin dispatch. On trip, raise EvalError::Cancelled with the current step number and the partial last_losses. mlpl-repl --connect binds Ctrl-C on the second press (within a short window) to a cancel POST.

TDD (Red/Green/Refactor):

1. RED tests:
   - crates/mlpl-serve/tests/cancel_tests.rs: cancel mid-train returns the partial loss curve; cancel mid-builtin returns promptly; double-cancel is idempotent.
   - crates/mlpl-eval/tests/interrupt_tests.rs: eval_program respects Interrupt at loop heads and pre-builtin checkpoints; non-cancelled programs unaffected.
   - apps/mlpl-repl/tests/connect_cancel_tests.rs: Ctrl-Ctrl-C within 2s window posts /cancel and prints partial last_losses.

2. GREEN:
   - crates/mlpl-eval/src/interrupt.rs (new module): Interrupt = Arc<AtomicBool>; helpers check_or_err() + EvalError::Cancelled { step: usize, partial_losses: Vec<f64> }.
   - Thread Interrupt through Environment via set_interrupt / clear_interrupt / interrupt() like the MetricSink pattern; check at top of for/train/repeat loops + before builtin dispatch.
   - crates/mlpl-serve: new cancel_handler that flips the session's interrupt bool; install Interrupt + MetricSink in eval_stream_handler so /cancel works against the in-flight stream.
   - apps/mlpl-repl/src/connect_repl.rs: SIGINT handler (ctrlc crate or signal-hook) that on second press within window posts /cancel; surface partial_losses.

3. REFACTOR: keep new modules within sw-checklist budgets; document the cancellation contract in contracts/serve-contract/sessions-and-eval.md.

Quality gates per /mw-cp: cargo test (workspace), cargo clippy --all-targets --all-features -- -D warnings, cargo fmt + check, markdown-checker on touched contracts, sw-checklist (baseline must hold).

Commit before agentrail complete. Push after commit.

Out of scope (later steps): viz storage endpoint (step 004), non-SVG viz cache (step 005), web REPL connect mode (Phase 4).