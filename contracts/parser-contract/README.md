# Parser Contract

## Purpose

Define the lexer and parser behavioral spec for MLPL. This contract
governs `mlpl-parser`, which transforms source text into tokens and
then into an AST. The parser does NOT depend on `mlpl-array`.

## Key Types and Concepts

### Token

A lexer output unit with a kind and a source span.

- `Token { kind: TokenKind, span: Span }`
- `TokenKind` variants: `IntLit`, `FloatLit`, `Ident`, punctuation,
  operators, `Newline`, `Eof`
- Comments are recognized and skipped (not emitted as tokens)

### Span

Source location type (defined in `mlpl-core`, used here).

- Byte offset range into the source string
- Used for error reporting and trace correlation

### Lexer

Transforms `&str` source into a sequence of tokens.

- `lex(source: &str) -> Result<Vec<Token>, ParseError>`
- Skips whitespace (except newlines, which may be significant)
- Skips comments
- Produces `Eof` as the final token

### AST (future)

Parser-owned syntax nodes. Array literals in the AST are represented
as parser-owned nodes (e.g., list of expression nodes), NOT as
`mlpl-array` types. This keeps parser context small.

### Record literals and field access (Saga 29 step 001)

- `Expr::RecordLit { fields: Vec<(String, Expr)>, span }` --
  `{ name1: expr1, name2: expr2, ... }`. Empty record `{}` is legal.
  Trailing comma is accepted. Field names must be idents. Duplicate
  field names error at parse time
  (`ParseError::DuplicateRecordField`).
- `Expr::FieldAccess { receiver: Box<Expr>, field: String, span }` --
  `expr.ident`. Binds tighter than every infix binop so `f(x).y + z`
  parses as `(f(x).y) + z`. Chained access `a.b.c` is left-
  associative.
- `TokenKind::Dot` -- single-character `.`. Distinct from float-
  literal decimal points: the float lexer only consumes
  `digit.digit`; a bare `.` at lexer fall-through becomes this token.
- Block-vs-record disambiguation is structural: `{ stmt; ... }`
  blocks only appear after the `repeat` / `train` / `for` /
  `experiment` / `device` keywords (their parsers consume the `{`
  directly via `parse_braced_body`). In any other position, `{`
  opens a record literal.

## Invariants

- Token kinds are stable once defined (do not renumber or reorder)
- Every token carries a valid `Span` into the original source
- Spans must not overlap
- Spans must cover the full source (no gaps except skipped whitespace/comments)
- Invalid input must produce explicit diagnostics with span info
- The lexer must not panic on any input
- `Eof` is always the last token

## Error Cases

Use explicit error variants local to `mlpl-parser`.

- `UnexpectedCharacter(char, Span)` -- character not valid in any token
- `UnterminatedString(Span)` -- if string literals are added later
- `InvalidNumber(Span)` -- malformed numeric literal
- `UnexpectedToken(TokenKind, Span)` -- parser-level, wrong token kind
- `DuplicateRecordField { name, span }` -- Saga 29 step 001: a
  record literal repeated a field name. Eval can assume names are
  unique because the parser rejects duplicates.

## What This Contract Does NOT Cover

- Semantic analysis or type checking
- Runtime evaluation of expressions
- Array storage or tensor operations (that is `mlpl-array`)
- Unicode syntax (ASCII-first; Unicode aliases are future)

## Open Questions

- Exact identifier syntax (leading alpha + alphanumeric? underscores?)
- Whether newlines are significant (statement separator vs whitespace)
- Array literal punctuation form: `[1, 2, 3]` vs space-separated
- When Unicode operator aliases enter the surface syntax
- Whether the parser produces a concrete or abstract syntax tree
