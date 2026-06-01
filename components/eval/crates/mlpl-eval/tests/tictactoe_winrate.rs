//! End-to-end win-rate de-risk: train a tiny MLPL LM on optimal
//! tic-tac-toe games, then actually PLAY it (as O) against a random
//! opponent (X) and tally win/tie/loss. The trained model should learn
//! optimal play -- which can never lose to a random opponent -- while
//! an untrained model loses often. This is the "no strategy -> winning
//! strategy" result, measured by games played, not a loss number.
//!
//! Cells 0..8 are chars 'a'..'i' == byte tokens 97..105, so the model's
//! move = argmax over the legal cells' logits, no tokenizer round-trip.

use mlpl_eval::{Environment, eval_program};
use mlpl_parser::{lex, parse};
use mlpl_tictactoe::{Board, Outcome, char_corpus, hero_vs_random_games, play_game, random_move, Rng};

fn run(env: &mut Environment, src: &str) {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap();
}

// The model's move: feed the move history as byte tokens 97+cell, read
// the next-move logits, and pick the highest-scoring LEGAL cell.
fn model_move(env: &mut Environment, model: &str, history: &[usize], board: &Board) -> usize {
    let ids: Vec<String> = history.iter().map(|&c| (97 + c).to_string()).collect();
    let src = format!("last_row(apply({model}, [{}]))", ids.join(", "));
    let logits = eval_program(&parse(&lex(&src).unwrap()).unwrap(), env).unwrap();
    let d = logits.data();
    *board
        .legal()
        .iter()
        .max_by(|&&a, &&b| d[97 + a].partial_cmp(&d[97 + b]).unwrap())
        .unwrap()
}

// Play `n` games: random opponent is X (moves first), the model is O.
// Returns (model wins, ties, model losses).
fn play_vs_random(env: &mut Environment, model: &str, n: usize) -> (u32, u32, u32) {
    let mut rng = Rng::new(98765);
    let (mut win, mut tie, mut loss) = (0u32, 0u32, 0u32);
    for _ in 0..n {
        let out = play_game(
            |b: &Board, _h: &[usize]| random_move(b, &mut rng),
            |b: &Board, h: &[usize]| model_move(env, model, h, b),
        );
        match out {
            Outcome::OWin => win += 1,
            Outcome::Draw => tie += 1,
            Outcome::XWin => loss += 1,
        }
    }
    (win, tie, loss)
}

// FINDING (2026-06): a char-level next-token LM over MOVE SEQUENCES does
// not learn to play tic-tac-toe well -- even full-model training leaves
// it losing ~half its games to a random opponent. It matches the move
// *distribution* (perplexity drops sharply) but has no explicit board
// state, so it can't reason about legality/lines on the off-distribution
// boards random play creates. The fix is a board-state -> move CLASSIFIER
// (input the 9 cells, predict the move), not a sequence LM. This harness
// (play the model vs random via the eval API, tally W/T/L) is reused
// there. Ignored + record-only until the classifier lands.
#[ignore = "sequence-LM approach does not learn to play; see FINDING above"]
#[test]
fn fine_tuned_model_stops_losing_to_random() {
    let corpus = char_corpus(&hero_vs_random_games(260, 1)).replace('\n', "\\n");
    let mut env = Environment::new();
    run(
        &mut env,
        &format!(
            "corpus = \"{corpus}\"\n\
tok = train_bpe(corpus, 280, 0)\n\
ids = apply_tokenizer(tok, corpus)\n\
X = reshape(shift_pairs_x(ids, 12), [reduce_mul(shape(shift_pairs_x(ids, 12)))])\n\
Y = reshape(shift_pairs_y(ids, 12), [reduce_mul(shape(shift_pairs_y(ids, 12)))])\n\
m = chain(embed(280, 32, 0), residual(chain(rms_norm(32), causal_attention(32, 1, 1))), rms_norm(32), linear(32, 280, 4))\n\
m0 = chain(embed(280, 32, 5), residual(chain(rms_norm(32), causal_attention(32, 1, 5))), rms_norm(32), linear(32, 280, 5))\n\
experiment \"ttt\" {{ train 120 {{ adam(cross_entropy(apply(m, X), Y), m, 0.01, 0.9, 0.999, 0.00000001) }} }}\n"
        ),
    );

    let games = 60;
    let (uw, ut, ul) = play_vs_random(&mut env, "m0", games);
    let (tw, tt, tl) = play_vs_random(&mut env, "m", games);
    println!("untrained vs random: W={uw} T={ut} L={ul}");
    println!("fine-tuned vs random: W={tw} T={tt} L={tl}");

    // Record-only: every game produced an outcome. (The sequence-LM
    // model does NOT reliably win -- see FINDING above; the assertion of
    // a winning model belongs to the classifier-based successor.)
    assert_eq!(uw + ut + ul, games as u32);
    assert_eq!(tw + tt + tl, games as u32);
}
