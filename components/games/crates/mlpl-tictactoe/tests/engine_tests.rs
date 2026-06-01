//! Engine contract: win detection, and that minimax play never loses
//! (optimal vs optimal is always a draw; optimal vs random never loses
//! -- the property the fine-tuned model should learn to imitate).

use mlpl_tictactoe::Outcome;
use mlpl_tictactoe::board::Board;
use mlpl_tictactoe::minimax::best_moves;
use mlpl_tictactoe::play::{Rng, play_game, random_move};

#[test]
fn winner_detects_a_row() {
    let mut b = Board::default();
    b.play(0, 1);
    b.play(1, 1);
    b.play(2, 1);
    assert_eq!(b.winner(), 1);
}

#[test]
fn optimal_vs_optimal_always_draws() {
    for seed in 0..50 {
        let mut rng = Rng::new(seed + 1);
        let optimal = |player: i8| {
            move |b: &Board, _h: &[usize]| {
                let mut bb = *b;
                best_moves(&mut bb, player)[0]
            }
        };
        // Deterministic optimal both sides -> always a draw.
        let out = play_game(optimal(1), optimal(-1));
        assert_eq!(out, Outcome::Draw, "seed {seed}");
        let _ = &mut rng;
    }
}

#[test]
fn optimal_o_never_loses_to_random_x() {
    let mut losses = 0;
    for seed in 0..200 {
        let mut rng = Rng::new(seed * 2 + 1);
        let out = play_game(
            |b: &Board, _h: &[usize]| random_move(b, &mut rng),
            |b: &Board, _h: &[usize]| {
                let mut bb = *b;
                best_moves(&mut bb, -1)[0]
            },
        );
        // O is optimal -> X (random) must never win.
        if out == Outcome::XWin {
            losses += 1;
        }
    }
    assert_eq!(losses, 0, "optimal O lost {losses} games to random X");
}
