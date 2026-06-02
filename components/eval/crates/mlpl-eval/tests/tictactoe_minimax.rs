//! Runs the MLPL-native alpha-beta tic-tac-toe minimax
//! (examples/tictactoe-minimax.mlpl) and checks the optimal values.
//! Exercises the language-correctness fixes end-to-end: per-call scope
//! (C1, recursion), multi-line array literals (C2, the incidence
//! matrix). Fast thanks to alpha-beta pruning (perf saga P-B); the
//! naive minimax took ~10.8s on the empty board.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

const SRC: &str = include_str!("../../../../../examples/tictactoe-minimax.mlpl");

fn run() -> Vec<f64> {
    let mut env = Environment::new();
    eval_program(&parse(&lex(SRC).unwrap()).unwrap(), &mut env)
        .unwrap()
        .data()
        .to_vec()
}

#[test]
fn mlpl_minimax_gives_optimal_values() {
    // Deep recursion through the AST interpreter needs more stack than a
    // debug test thread's default; run on a roomy stack.
    let r = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
    // Near-won board with X to move -> 1; empty board -> 0 (the
    // perfect-play draw). Empty was the bug: pre-C1 it returned 1.
    assert_eq!(r, vec![1.0, 0.0]);
}
