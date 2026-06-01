//! Generate a training corpus of optimal games. Both sides play
//! minimax-optimal with random tie-breaking (for variety); each game is
//! a sequence of cells. Cells 0..8 encode as chars 'a'..'i' so the
//! games train through MLPL's text LM pipeline with no special
//! tokenizer (char 'a'+c is byte 97+c).

use crate::board::Board;
use crate::minimax::best_moves;
use crate::play::{Rng, random_move};

/// Play `n` optimal-vs-optimal games (random tie-break), returning each
/// game's move sequence.
pub fn optimal_games(n: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let mut b = Board::default();
            let mut moves = Vec::new();
            let mut player = 1i8;
            while !b.terminal() {
                let opts = best_moves(&mut b, player);
                let c = opts[rng.below(opts.len())];
                b.play(c, player);
                moves.push(c);
                player = -player;
            }
            moves
        })
        .collect()
}

/// Games where one side plays minimax-optimal (random tie-break) and
/// the other plays randomly; the optimal "hero" alternates sides across
/// games. Unlike [`optimal_games`] (all draws), these visit the
/// off-distribution positions a random opponent creates AND label the
/// optimal response -- so a model trained on them learns to actually
/// beat random play, not merely draw against perfect play.
pub fn hero_vs_random_games(n: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let hero: i8 = if i % 2 == 0 { 1 } else { -1 };
            let mut b = Board::default();
            let mut moves = Vec::new();
            let mut player = 1i8;
            while !b.terminal() {
                let c = if player == hero {
                    let opts = best_moves(&mut b, player);
                    opts[rng.below(opts.len())]
                } else {
                    random_move(&b, &mut rng)
                };
                b.play(c, player);
                moves.push(c);
                player = -player;
            }
            moves
        })
        .collect()
}

/// Join games into a newline-separated char corpus (cell c -> 'a'+c).
pub fn char_corpus(games: &[Vec<usize>]) -> String {
    let mut s = String::new();
    for g in games {
        for &c in g {
            s.push((b'a' + c as u8) as char);
        }
        s.push('\n');
    }
    s
}
