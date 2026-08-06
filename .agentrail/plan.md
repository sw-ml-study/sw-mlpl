# Saga: apl2-idioms-expunge
User direction 2026-08-06: (1) APL )ERASE / quad-EX parity --
`expunge(name | [names])` builtin (1 = name free afterwards,
idempotent; 0 = malformed name; clears every value table, u:
functions, and their @test registry rows) plus the `:erase`
REPL command (space-separated names, :fns lineage). (2)
docs/apl2-idioms.mlpl: an EXECUTABLE Rosetta document -- APL2
expressions as Unicode comments beside equivalent MLPL, with
not-yet-expressible idioms explicitly marked; loads in the web
editor and runs clean.
## Steps
1. expunge -- builtin + :erase command + usage-table row + docs
   rows. TDD.
2. apl2-idioms -- the runnable mapping document + a test that
   executes it; web-editor loadability check.
3. close -- rebuilds (repl/serve/pages), wiki, q-and-a, --done.
