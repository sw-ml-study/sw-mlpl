# components/lang-syntax/ migration + split (saga 54)

Move + DECOMPOSE the lexer/parser/macro/lower-rs family into a
new `components/lang-syntax/` workspace. Apply the move-AND-split
principle: every crowded crate must be split into sparse siblings
inside the component, not just moved.

## Crowded crates to split

- `mlpl-lexer`: 5 modules + 7-fn lexer.rs WARN + 6-fn lex_util.rs WARN +
  3 long-fn WARNs (next_token 35, lex_ident 32, describe_kind 39).
  Plan to split into ~6 sibling crates by lexing concern.
- `mlpl-parser`: 6 modules at limit + ast_fmt 7-fn WARN + parser.rs
  5-fn WARN + parser.rs 415-line file WARN + fmt() 49-line WARN.
  Plan to split into ~4 sibling crates by parsing concern.

## Step plan

1. **scaffold**: create `components/lang-syntax/` workspace.
2. **move-macro**: move mlpl-macro (small, leaf).
3. **move-lower-rs**: move mlpl-lower-rs (depends on parser only).
4. **move-and-split-lexer**: move mlpl-lexer into the new component,
   then decompose it into sibling crates: mlpl-lexer-token (types),
   mlpl-lexer-error (ParseError + describe_kind), mlpl-lex-string
   (string literal + utf-8), mlpl-lex-number, mlpl-lex-punct
   (whitespace + single_char + builtin_ref), mlpl-lex-ident
   (keyword recognition + identifier), and mlpl-lexer (Lexer driver
   + entry point) calling all the helpers.
5. **move-and-split-parser**: move mlpl-parser into the new
   component, then decompose: mlpl-parser-ast (AST types),
   mlpl-parser-fmt (Display impls), mlpl-parser-stmts
   (statement parsing), mlpl-parser-records (record literal +
   field access), mlpl-parser (main parser entry).
6. **close**: sw-checklist delta, language-status update.

## Expected sw-checklist deltas

Each split should retire WARNs through structural decomposition:
- lexer family: ~5 WARNs retired (lex_util 6-fn, lexer.rs 7-fn,
  describe_kind 39, next_token 35, lex_ident 32)
- parser family: ~4 WARNs retired (ast_fmt 7-fn, parser.rs 5-fn,
  parser.rs 415-line, fmt 49)

Plus any structural FAILs if module counts go over budget for the
mlpl-parser/lexer crates that currently FAIL. (They're at WARN now,
not FAIL, so no FAILs to retire from this saga's primary work.)
