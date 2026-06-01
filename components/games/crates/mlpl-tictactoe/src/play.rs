//! Self-play harness + a tiny deterministic RNG. `play_game` alternates
//! two move-pickers (X first, then O), each given the board and the
//! move history so far, and returns the [`Outcome`].

use crate::board::{Board, Outcome};

/// Deterministic xorshift64* RNG (no external `rand` dependency).
pub struct Rng(pub u64);

impl Rng {
    /// Seed the RNG (forced odd + nonzero).
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    /// A value in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) % n as u64) as usize
    }
}

/// Pick a uniformly-random legal move.
pub fn random_move(b: &Board, rng: &mut Rng) -> usize {
    let m = b.legal();
    m[rng.below(m.len())]
}

/// Play one game: `x` picks for player +1 (moves first), `o` for -1.
/// Each picker receives `(&Board, &[usize] history)` and returns a
/// legal cell. Returns the outcome from X's perspective.
pub fn play_game<X, O>(mut x: X, mut o: O) -> Outcome
where
    X: FnMut(&Board, &[usize]) -> usize,
    O: FnMut(&Board, &[usize]) -> usize,
{
    let mut board = Board::default();
    let mut history = Vec::new();
    let mut player = 1i8;
    while !board.terminal() {
        let c = if player == 1 {
            x(&board, &history)
        } else {
            o(&board, &history)
        };
        board.play(c, player);
        history.push(c);
        player = -player;
    }
    match board.winner() {
        1 => Outcome::XWin,
        -1 => Outcome::OWin,
        _ => Outcome::Draw,
    }
}
