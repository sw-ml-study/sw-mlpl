//! Parser for MLPL source code. Lexer and token types live in
//! the `mlpl-lexer` crate.

mod ast;
mod ast_fmt;
mod ast_fmt_compound;
mod parser;
mod record_parser;
mod stmts;

pub use ast::{BinOpKind, Expr, TensorCtorKind};
pub use mlpl_lexer::{ParseError, Token, TokenKind, lex};
pub use parser::parse;
