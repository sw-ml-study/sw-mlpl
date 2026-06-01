//! 3x3 board: cells 0..8, X = +1, O = -1, empty = 0.

/// Game result from X's perspective.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    /// X (the first player) won.
    XWin,
    /// O (the second player) won.
    OWin,
    /// All nine cells filled with no three-in-a-row.
    Draw,
}

const LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8], // rows
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8], // cols
    [0, 4, 8],
    [2, 4, 6], // diagonals
];

/// A tic-tac-toe board. `cells[i]` is +1 (X), -1 (O), or 0 (empty).
#[derive(Clone, Copy, Default)]
pub struct Board {
    /// The nine cells in row-major order.
    pub cells: [i8; 9],
}

impl Board {
    /// `+1` if X has a line, `-1` if O does, `0` otherwise.
    pub fn winner(&self) -> i8 {
        for l in LINES {
            let s = self.cells[l[0]] + self.cells[l[1]] + self.cells[l[2]];
            if s == 3 {
                return 1;
            }
            if s == -3 {
                return -1;
            }
        }
        0
    }

    /// Indices of the still-empty cells.
    pub fn legal(&self) -> Vec<usize> {
        (0..9).filter(|&c| self.cells[c] == 0).collect()
    }

    /// True once the game is over (someone won or the board is full).
    pub fn terminal(&self) -> bool {
        self.winner() != 0 || self.cells.iter().all(|&c| c != 0)
    }

    /// Place `player` (+1 / -1) on cell `c`.
    pub fn play(&mut self, c: usize, player: i8) {
        self.cells[c] = player;
    }
}
