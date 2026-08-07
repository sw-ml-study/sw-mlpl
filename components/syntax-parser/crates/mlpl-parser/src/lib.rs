//! Parser for MLPL source code. AST types live in `mlpl-parser-ast`,
//! lexer and token types live in `mlpl-lexer`. This crate is the
//! orchestrator: it owns the `Parser` struct and the `parse` entry
//! point, with statement and record-literal parsing delegated to
//! sibling modules.

mod parser;
mod record_parser;
mod stmts;
mod token_stream;

pub use mlpl_lexer::{ParseError, Token, TokenKind, lex};
pub use mlpl_parser_ast::{BinOpKind, Expr, TensorCtorKind};
pub use parser::parse;
