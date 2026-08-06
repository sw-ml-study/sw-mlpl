# Saga: apl2-hof
User direction 2026-08-07 (order: this, then combinator-birds).
Higher-order builtins over function references, now cheap on
the callable machinery: each (APL f-umlaut / BQN modifier),
table (APL jot-dot / BQN table), atop and over (BQN
composition, immediate application). No function VALUES are
produced -- that is combinator-birds' Partial (see
docs/combinators-research.txt).
## Steps
1. each-table -- each(f, v): elementwise ref application, shape
   preserved, scalar in/out (v1); table(f, a, b): [m]x[n] ->
   [m, n] outer product over f. u: and builtin refs. TDD.
2. atop-over -- atop(f, g, x...) = f(g(x...)); over(f, g, x, y)
   = f(g(x), g(y)); docs rows + idioms-doc section + catalog;
   glossary + README pin.
3. close -- demos touch-up, rebuild repl/serve/pages + deploy,
   wiki row, mlplunit-adjacent none; queue advance to
   combinator-birds; --done.
