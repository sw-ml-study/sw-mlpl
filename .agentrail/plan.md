# Saga: apl2-blocks
APL2 re-ranking parity on flat arrays (user direction
2026-08-06): generalized dyadic transpose + blocked rank-4
disp, so `disp(reshape(range(81), [3,3,3,3]))` renders a 3x3
grid of boxed 3x3 matrices (APL2's DISPLAY of enclose[3 4]),
and `transpose_axes(reshape(B, [3,3,3,3]), [1,3,2,4])` makes a
9x9 board block-major. True enclose/nested arrays are a
separate queued design program. Analysis: docs/q-and-a.md
2026-08-06 evening. (gen-state-kv-cache paused after step 001;
resume at gen-controls.)
## Steps
1. transpose-axes -- transpose_axes(x, perm) axis permutation
   for any rank; TDD (parity with transpose on rank-2, Sudoku
   block extraction, error handling).
2. blocked-disp -- disp renders rank-3/4 arrays as an outer
   grid of boxed inner matrices (native + web); TDD.
3. demo-and-docs -- Sudoku-blocks / iota-81 demo, catalog +
   lang-reference + glossary rows, pages rebuild, close.
