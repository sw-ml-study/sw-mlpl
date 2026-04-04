# Parser Contract

## Purpose

Define the first parser and lexer contract for MLPL.

## Initial scope
- ASCII-first lexing
- numeric literals
- core punctuation
- source spans
- explicit lexer errors

## Invariants
- token kinds must be stable
- spans must identify source positions deterministically
- invalid input must produce explicit diagnostics

## Open questions
- exact identifier syntax
- earliest array literal punctuation form
- when Unicode aliases enter the surface syntax
