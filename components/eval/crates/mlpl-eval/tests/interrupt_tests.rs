//! Saga 21.5 step 003: cooperative cancellation in `eval_program`.
//!
//! The evaluator threads an `Interrupt` token through `Environment`.
//! When the token is tripped, every loop head (`for`, `train`,
//! `repeat`) and every pre-builtin checkpoint raises
//! `EvalError::Cancelled`. `train` enriches the bubbled error with
//! the current iteration index plus the partial loss curve so
//! clients can render what completed before the cancel landed;
//! `for` / `repeat` / pre-builtin sites raise an unenriched
//! `Cancelled` (empty `partial_losses`).
//!
//! Non-cancelled programs are unaffected: the `Interrupt` defaults
//! to `false` and stays that way unless something explicitly trips
//! it.
//!
//! The tests use `Environment::set_interrupt` to install a shared
//! `Interrupt` and `Interrupt::set()` (called from a background
//! thread or pre-eval) to simulate the server-side `/cancel` flip.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use mlpl_eval::{Environment, EvalError, Interrupt, MetricSink, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(program: &str, env: &mut Environment) -> Result<mlpl_eval::Value, EvalError> {
    let tokens = lex(program).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    eval_program_value(&stmts, env)
}

/// A program with no `train` / `for` / `repeat` and no `Interrupt`
/// installed is unaffected -- baseline parity check.
#[test]
fn no_interrupt_no_change() {
    let mut env = Environment::new();
    let v = run("iota(5) + 1", &mut env).unwrap();
    assert!(format!("{v}").contains('4'));
}

/// An installed-but-not-tripped `Interrupt` does not affect
/// evaluation. The gradual-additivity rule applies: the server can
/// install the token unconditionally without changing semantics.
#[test]
fn untripped_interrupt_does_not_block() {
    let mut env = Environment::new();
    env.set_interrupt(Interrupt::new());
    let v = run("train 3 { step + 1 }", &mut env).unwrap();
    assert!(format!("{v}").contains('3'));
}

/// `Interrupt::set()` BEFORE eval runs raises `Cancelled` at the
/// very first loop head -- step 0, empty `partial_losses` (no
/// iteration completed).
#[test]
fn cancel_before_train_raises_at_step_0() {
    let mut env = Environment::new();
    let intr = Interrupt::new();
    intr.set();
    env.set_interrupt(intr);
    let err = run("train 5 { step + 1 }", &mut env).unwrap_err();
    match err {
        EvalError::Cancelled {
            step,
            partial_losses,
        } => {
            assert_eq!(step, 0, "expected step 0, got {step}");
            assert!(
                partial_losses.is_empty(),
                "expected empty losses, got {partial_losses:?}"
            );
        }
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

/// Sink that flips the interrupt on the 3rd metric emission so the
/// 4th iteration's loop-head check trips. The bubbled `Cancelled`
/// carries `step = 3` and the 3 completed losses.
#[derive(Debug)]
struct CancelAfter {
    seen: Arc<AtomicUsize>,
    intr: Interrupt,
    cutoff: usize,
}

impl MetricSink for CancelAfter {
    fn emit(&self, _name: &str, _step: usize, _value: f64) {
        let prev = self.seen.fetch_add(1, Ordering::SeqCst);
        if prev + 1 >= self.cutoff {
            self.intr.set();
        }
    }
}

#[test]
fn cancel_mid_train_returns_partial_losses() {
    let mut env = Environment::new();
    let intr = Interrupt::new();
    let seen = Arc::new(AtomicUsize::new(0));
    env.set_metric_sink(Arc::new(CancelAfter {
        seen: seen.clone(),
        intr: intr.clone(),
        cutoff: 3,
    }));
    env.set_interrupt(intr);
    let err = run("train 10 { loss_metric = step + 0.5 }", &mut env).unwrap_err();
    match err {
        EvalError::Cancelled {
            step,
            partial_losses,
        } => {
            assert_eq!(
                step, 3,
                "expected to trip at step 3 (after 3 completed iters)"
            );
            assert_eq!(partial_losses.len(), 3, "{partial_losses:?}");
            for (i, v) in partial_losses.iter().enumerate() {
                let expected = i as f64 + 0.5;
                assert!(
                    (*v - expected).abs() < 1e-9,
                    "loss[{i}] = {v}, expected {expected}"
                );
            }
        }
        other => panic!("expected Cancelled, got {other:?}"),
    }
    // The session's env should expose the same partial loss curve
    // via `last_losses` so the client can inspect it after the cancel.
    let losses = env.get("last_losses").expect("last_losses bound");
    assert_eq!(losses.data().len(), 3);
}

/// Cancelling at a `repeat` loop head also raises `Cancelled` -- with
/// `step` equal to the iteration index where the trip was caught,
/// and an empty `partial_losses` (only `train` accumulates losses).
#[test]
fn cancel_in_repeat_raises_at_loop_head() {
    let mut env = Environment::new();
    let intr = Interrupt::new();
    intr.set();
    env.set_interrupt(intr);
    let err = run("repeat 5 { iota(3) }", &mut env).unwrap_err();
    match err {
        EvalError::Cancelled {
            step,
            partial_losses,
        } => {
            assert_eq!(step, 0);
            assert!(partial_losses.is_empty());
        }
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

/// Cancelling at a `for` loop head also raises `Cancelled`. Uses an
/// already-tripped interrupt so the very first iteration head trips.
#[test]
fn cancel_in_for_raises_at_loop_head() {
    let mut env = Environment::new();
    let intr = Interrupt::new();
    intr.set();
    env.set_interrupt(intr);
    let err = run("for x in iota(5) { x + 1 }", &mut env).unwrap_err();
    assert!(
        matches!(err, EvalError::Cancelled { .. }),
        "expected Cancelled, got {err:?}"
    );
}

/// Cancelling at a pre-builtin checkpoint trips even outside any
/// loop. A `Cancel` set just before a builtin dispatch is observed
/// at the head of the next builtin call. The eval returns with
/// `Cancelled { step: 0, partial_losses: [] }` -- no enclosing loop
/// to enrich.
#[test]
fn cancel_at_pre_builtin_checkpoint() {
    let mut env = Environment::new();
    let intr = Interrupt::new();
    intr.set();
    env.set_interrupt(intr);
    // `reduce_add` is a builtin; the pre-dispatch check trips
    // before the call runs.
    let err = run("reduce_add(iota(100))", &mut env).unwrap_err();
    assert!(
        matches!(err, EvalError::Cancelled { .. }),
        "expected Cancelled, got {err:?}"
    );
}

/// The `Interrupt` is shareable across threads -- the server's
/// `/cancel` handler flips the bool from a different thread than
/// the eval task.
#[test]
fn interrupt_set_from_another_thread() {
    let intr = Interrupt::new();
    let intr_clone = intr.clone();
    let handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));
        intr_clone.set();
    });
    while !intr.is_set() {
        thread::sleep(Duration::from_millis(5));
    }
    handle.join().unwrap();
    assert!(intr.is_set());
}

/// `Interrupt::reset()` clears a previously-set token so the same
/// `Session` can run another eval after a cancel completes.
#[test]
fn interrupt_reset_clears_the_bool() {
    let intr = Interrupt::new();
    intr.set();
    assert!(intr.is_set());
    intr.reset();
    assert!(!intr.is_set());
}
