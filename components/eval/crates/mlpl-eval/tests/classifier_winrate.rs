//! The fix: a board-state -> move CLASSIFIER (not a move-sequence LM).
//! Input the 9-cell board (one-hot empty/mine/theirs, 27 features),
//! predict the move (9 logits); train with cross_entropy on the
//! complete (board, optimal-move) policy dataset. Then PLAY the trained
//! classifier vs a random opponent and tally W/T/L. A correct optimal
//! policy never loses to random -- so the trained model should stop
//! losing, while an untrained one loses often.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};
use mlpl_tictactoe::{Board, Outcome, Rng, encode, play_game, policy_dataset, random_move};

fn run(env: &mut Environment, src: &str) {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap();
}

fn eval_arr(env: &mut Environment, src: &str) -> Vec<f64> {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env)
        .unwrap()
        .data()
        .to_vec()
}

// The model plays as O (-1): encode the board from its perspective,
// score all 9 cells, pick the highest-scoring LEGAL one.
fn model_move(env: &mut Environment, model: &str, board: &Board) -> usize {
    let enc = encode(&board.cells, -1).to_vec();
    env.set(
        "__b".into(),
        DenseArray::new(Shape::new(vec![1, 27]), enc).unwrap(),
    );
    let logits = eval_arr(env, &format!("apply({model}, __b)"));
    *board
        .legal()
        .iter()
        .max_by(|&&a, &&b| logits[a].partial_cmp(&logits[b]).unwrap())
        .unwrap()
}

fn play_vs_random(env: &mut Environment, model: &str, n: usize) -> (u32, u32, u32) {
    let mut rng = Rng::new(98765);
    let (mut w, mut t, mut l) = (0, 0, 0);
    for _ in 0..n {
        let out = play_game(
            |b: &Board, _h: &[usize]| random_move(b, &mut rng),
            |b: &Board, _h: &[usize]| model_move(env, model, b),
        );
        match out {
            Outcome::OWin => w += 1,
            Outcome::Draw => t += 1,
            Outcome::XWin => l += 1,
        }
    }
    (w, t, l)
}

// ~100s (4520-position dataset, 500 Adam steps in the AST interpreter):
// run explicitly with `--ignored`, not in the default suite.
#[test]
#[ignore]
fn classifier_stops_losing_to_random() {
    let (x, y, n) = policy_dataset();
    let mut env = Environment::new();
    env.set(
        "X".into(),
        DenseArray::new(Shape::new(vec![n, 27]), x).unwrap(),
    );
    env.set("Y".into(), DenseArray::new(Shape::new(vec![n]), y).unwrap());
    run(
        &mut env,
        "m = chain(linear(27, 128, 0), relu_layer(), linear(128, 9, 1))\n\
m0 = chain(linear(27, 128, 5), relu_layer(), linear(128, 9, 6))\n\
experiment \"ttt\" { train 500 { adam(cross_entropy(apply(m, X), Y), m, 0.05, 0.9, 0.999, 0.00000001) } }\n",
    );
    let acc = eval_arr(&mut env, "mean(eq(argmax(apply(m, X), 1), Y))")[0];
    println!("positions={n}  train next-move accuracy={acc:.3}");

    let games = 100;
    let (uw, ut, ul) = play_vs_random(&mut env, "m0", games);
    let (tw, tt, tl) = play_vs_random(&mut env, "m", games);
    println!("untrained vs random:  W={uw} T={ut} L={ul}");
    println!("classifier vs random: W={tw} T={tt} L={tl}");

    // The observable before/after: training a board-state policy classifier
    // strictly cuts losses and raises wins vs an untrained net.
    assert!(
        tl < ul,
        "classifier should lose fewer than untrained ({tl} vs {ul})"
    );
    assert!(
        tw > uw,
        "classifier should win more than untrained ({tw} vs {uw})"
    );
}
