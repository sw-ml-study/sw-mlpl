//! `play_vs_random(model, n)` builtin: plays the named model as O vs a
//! random opponent for n games and returns outcome counts `[losses,
//! ties, wins]`. Here we just check the contract (3 counts that sum to
//! n) with an untrained model -- the win-rate quality lives in the
//! `classifier_winrate` test.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Vec<f64> {
    let mut env = Environment::new();
    eval_program(&parse(&lex(src).unwrap()).unwrap(), &mut env)
        .unwrap()
        .data()
        .to_vec()
}

#[test]
fn play_vs_random_returns_outcome_counts() {
    let counts = eval(
        "m = chain(linear(27, 9, 0))\n\
play_vs_random(m, 30)\n",
    );
    assert_eq!(counts.len(), 3, "expected [losses, ties, wins]");
    let total: f64 = counts.iter().sum();
    assert_eq!(total, 30.0, "every game has exactly one outcome");
}

#[test]
fn play_vs_random_unknown_model_errors() {
    let mut env = Environment::new();
    let prog = parse(&lex("play_vs_random(nope, 5)").unwrap()).unwrap();
    assert!(eval_program(&prog, &mut env).is_err());
}
