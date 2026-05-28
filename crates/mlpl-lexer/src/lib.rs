//! Lexer and token types for MLPL source code, extracted from
//! mlpl-parser.

mod error;
mod lex_util;
mod lexer;
mod token;

pub use error::{ParseError, describe_kind};
pub use lexer::lex;
pub use token::{Token, TokenKind};
