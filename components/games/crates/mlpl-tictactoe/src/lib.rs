//! Tic-tac-toe environment for the fine-tuning demo.
//!
//! - [`board`]: the 3x3 board + win detection.
//! - [`minimax`]: optimal play (and the optimal-move set, for variety).
//! - [`play`]: a self-play harness parameterized by two move-pickers,
//!   plus a tiny deterministic RNG.
//! - [`corpus`]: generate a training corpus of optimal games encoded as
//!   char sequences (cells 0..8 -> 'a'..'i').
//!
//! The demo uses this to show a tiny model move from no strategy
//! (loses to a random opponent) to a winning one (never loses) after
//! LoRA fine-tuning on the optimal-game corpus.

pub mod board;
pub mod corpus;
pub mod minimax;
pub mod play;

pub use board::{Board, Outcome};
pub use corpus::{char_corpus, hero_vs_random_games, optimal_games};
pub use minimax::best_moves;
pub use play::{Rng, play_game, random_move};
