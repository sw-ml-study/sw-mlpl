//! Win-rate arena: play a model (as O) against a random opponent and
//! tally outcomes. The caller injects the model's move policy, so this
//! crate stays free of any eval/model dependency (the `play_vs_random`
//! builtin lives in `mlpl-eval` and supplies the forward pass).

use crate::Board;
use crate::board::Outcome;
use crate::play::{Rng, play_game, random_move};

/// Play `n` games of `model_move` (as O) against a random opponent
/// (as X, moving first) and tally outcome counts `[losses, ties,
/// wins]` from the model's (O's) perspective.
pub fn play_vs_random_counts<F>(n: usize, seed: u64, mut model_move: F) -> [u32; 3]
where
    F: FnMut(&Board) -> usize,
{
    let mut rng = Rng::new(seed);
    let mut counts = [0u32; 3];
    for _ in 0..n {
        let out = play_game(|b, _h| random_move(b, &mut rng), |b, _h| model_move(b));
        counts[match out {
            Outcome::XWin => 0,
            Outcome::Draw => 1,
            Outcome::OWin => 2,
        }] += 1;
    }
    counts
}
