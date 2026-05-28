# Split lang-syntax (11 crates) into sparse sub-components (saga 68)

- syntax-tokens (2):  mlpl-lexer-token, mlpl-lexer-error
- syntax-lex (4):     mlpl-lex-string, mlpl-lex-number, mlpl-lex-punct, mlpl-lex-ident
- syntax-lexer (1):   mlpl-lexer
- syntax-parser (2):  mlpl-parser, mlpl-parser-ast
- syntax-codegen (2): mlpl-macro, mlpl-lower-rs
