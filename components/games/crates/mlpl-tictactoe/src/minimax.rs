//! Minimax: optimal play. `score` is the game-theoretic value from X's
//! perspective (+1 X wins, -1 O wins, 0 draw) under perfect play;
//! `best_moves` returns every move achieving that value (so a caller
//! can break ties randomly for variety).

use crate::board::Board;

/// Minimax value of `b` with `player` (+1 / -1) to move.
pub fn score(b: &mut Board, player: i8) -> i8 {
    let w = b.winner();
    if w != 0 || b.terminal() {
        return w;
    }
    let mut best = -2 * player; // worse than any real score for `player`
    for c in b.legal() {
        b.play(c, player);
        let s = score(b, -player);
        b.play(c, 0);
        best = if player == 1 {
            best.max(s)
        } else {
            best.min(s)
        };
    }
    best
}

/// All moves for `player` that achieve the optimal minimax value.
pub fn best_moves(b: &mut Board, player: i8) -> Vec<usize> {
    let scored: Vec<(usize, i8)> = b
        .legal()
        .into_iter()
        .map(|c| {
            b.play(c, player);
            let s = score(b, -player);
            b.play(c, 0);
            (c, s)
        })
        .collect();
    let best = scored.iter().map(|&(_, s)| s).fold(-2 * player, |a, s| {
        if player == 1 { a.max(s) } else { a.min(s) }
    });
    scored
        .into_iter()
        .filter(|&(_, s)| s == best)
        .map(|(c, _)| c)
        .collect()
}
