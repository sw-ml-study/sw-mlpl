//! B6 event-loop gate #1 spike: can the interpreter `Environment` move
//! to a worker thread and eval there? The browser-model design runs
//! MLPL on a spawned worker while winit owns the main thread, which
//! requires `Environment` (and `Value`) to be `Send`.

use mlpl_eval::{Environment, Value};

fn assert_send<T: Send>() {}
fn assert_clone<T: Clone>() {}

#[test]
fn environment_and_value_are_send() {
    assert_send::<Environment>();
    assert_send::<Value>();
    // Environment must stay Clone: downstream consumers (the demo-*
    // extension repos) depend on it. A port Receiver is not Clone, so
    // it is held behind Arc<Mutex> for exactly this reason.
    assert_clone::<Environment>();
    assert_clone::<Value>();
}

#[test]
fn eval_runs_on_a_spawned_worker_thread() {
    // Build the env on the worker, eval there, hand back a scalar.
    let handle = std::thread::spawn(|| {
        let mut env = Environment::new();
        let tokens = mlpl_parser::lex("reduce_add(range(7))").expect("lex");
        let stmts = mlpl_parser::parse(&tokens).expect("parse");
        match mlpl_eval::eval_program_value(&stmts, &mut env).expect("eval") {
            Value::Array(a) => a.data()[0],
            other => panic!("expected scalar, got {other:?}"),
        }
    });
    assert_eq!(handle.join().expect("worker join"), 21.0);
}

#[test]
fn env_built_on_main_moves_into_a_worker() {
    // The harder case: construct on the main thread, MOVE to the worker.
    let mut env = Environment::new();
    // seed a binding on main
    let toks = mlpl_parser::lex("x = range(5)").expect("lex");
    let stmts = mlpl_parser::parse(&toks).expect("parse");
    mlpl_eval::eval_program_value(&stmts, &mut env).expect("seed");
    let handle = std::thread::spawn(move || {
        let toks = mlpl_parser::lex("reduce_add(x)").expect("lex");
        let stmts = mlpl_parser::parse(&toks).expect("parse");
        match mlpl_eval::eval_program_value(&stmts, &mut env).expect("eval") {
            Value::Array(a) => a.data()[0],
            other => panic!("expected scalar, got {other:?}"),
        }
    });
    assert_eq!(handle.join().expect("worker join"), 10.0);
}
