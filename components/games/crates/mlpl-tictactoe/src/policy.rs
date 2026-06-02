//! Board-state policy dataset for the classifier model: every reachable
//! non-terminal position, encoded from the to-move player's perspective,
//! paired with its optimal move. This is the right representation for
//! learning a game policy (a move-sequence LM has no explicit board
//! state and fails to play -- see the eval-side win-rate finding).

use crate::board::Board;
use crate::minimax::best_moves;
use std::collections::HashSet;

/// One-hot encode a board from the to-move player's perspective: each of
/// the 9 cells contributes 3 features `[empty, mine, theirs]`, for 27
/// total. Normalizing by side-to-move makes the policy side-agnostic.
pub fn encode(cells: &[i8; 9], to_move: i8) -> [f64; 27] {
    let mut v = [0.0; 27];
    for (c, &cell) in cells.iter().enumerate() {
        let slot = if cell == 0 {
            0
        } else if cell == to_move {
            1
        } else {
            2
        };
        v[3 * c + slot] = 1.0;
    }
    v
}

/// Enumerate every reachable non-terminal position over the full legal
/// game tree (deduped by `(cells, to_move)`) and pair each with the
/// deterministic optimal move. Returns `(flat boards [N*27], moves
/// [N], N)` -- the complete supervised policy.
pub fn policy_dataset() -> (Vec<f64>, Vec<f64>, usize) {
    let mut seen = HashSet::new();
    let (mut x, mut y) = (Vec::new(), Vec::new());
    let mut stack = vec![([0i8; 9], 1i8)];
    while let Some((cells, p)) = stack.pop() {
        let mut b = Board { cells };
        if b.winner() != 0 || b.legal().is_empty() {
            continue;
        }
        if seen.insert((cells, p)) {
            x.extend_from_slice(&encode(&cells, p));
            y.push(best_moves(&mut b, p)[0] as f64);
        }
        for c in b.legal() {
            let mut next = cells;
            next[c] = p;
            stack.push((next, -p));
        }
    }
    let n = y.len();
    (x, y, n)
}
