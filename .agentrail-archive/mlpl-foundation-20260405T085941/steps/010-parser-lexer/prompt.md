Implement the MLPL lexer in mlpl-parser. This is task PARSER-LEX-001.

Reference docs/syntax-core-v1.md for the token set.

Implement in crates/mlpl-parser/src/lexer.rs:
1. Token enum with variants for:
   - IntLit(i64), FloatLit(f64)
   - Ident(String)
   - LParen, RParen, LBracket, RBracket
   - Comma, Equals, Semicolon
   - Plus, Minus, Star, Slash
   - Newline, Eof
   - Comment (skipped, not emitted)
2. Each token carries a Span (from mlpl-core)
3. Lexer struct that takes &str input and produces Vec<Token> or iterator
4. Handle: whitespace skipping, comment skipping, negative number vs minus operator

Error type in crates/mlpl-parser/src/error.rs:
1. ParseError enum with at least LexError variant
2. Include span information in errors

Write TDD-style:
- Test single tokens: integer, float, identifier, each punctuation
- Test multi-token expressions: "1 + 2", "[1, 2, 3]", "reshape(x, [2, 2])"
- Test comments are skipped
- Test error cases: invalid characters
- Test spans are correct

Verify:
- cargo test -p mlpl-parser passes
- cargo clippy -p mlpl-parser -- -D warnings passes

Allowed directories: crates/mlpl-parser/, contracts/parser-contract/
Do NOT import anything from mlpl-array.