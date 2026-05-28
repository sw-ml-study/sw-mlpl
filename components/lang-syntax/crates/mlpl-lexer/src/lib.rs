//! Top-level lexer orchestrator. Drives a stateful `Lexer` over
//! the source bytes and delegates per-token-kind to the per-concern
//! sibling crates. Re-exports the public lexer API so downstream
//! consumers can keep using `mlpl_lexer::{Token, TokenKind,
//! ParseError, describe_kind, lex}` after the split.

mod lexer;

pub use lexer::lex;
pub use mlpl_lexer_error::{ParseError, describe_kind};
pub use mlpl_lexer_token::{Token, TokenKind};
