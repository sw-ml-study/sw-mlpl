# reasoning-from-scratch-numerics

Cheap-core-first fixes for the ../reasoning-from-scratch findings
(RS1-RS12 in docs/sw-mlpl-findings.md). CORE = things a library/extension
cannot provide (autodiff rules, tensor primitives, dtypes, lexer syntax,
backend perf). LIBRARY/EXTENSION rows are handed to downstream and do not
gate these steps. Every step is TDD (RED gradcheck/lex test first) and
holds sw-checklist at or below baseline.

## Steps

1. unary-diff-sqrt-sin-cos-pow -- add tape UnaryKind backward rules for
   sqrt, sin, cos, pow (exp/log/sigmoid already present). RS1. TDD:
   finite-difference gradcheck per op; verify eager values unchanged.
2. sci-notation-literals -- lexer accepts 1e-4, 1.5e3, 2E-10, 6.02e23.
   RS5. TDD: lexer + eval tests; no regression on existing number/range
   forms.
3. matmul-rank3-clean-error -- replace the rank-3 matmul .expect() panic
   with a structured shape error (matmul analogue of F19); document 2-D
   scope. RS4. TDD: rank-3 input errors cleanly in and out of grad.
4. softmax-axis-in-grad -- thread the axis argument into the existing tape
   softmax and support rank-3; softmax(a, axis) differentiates. RS2. TDD:
   axis-param grad matches a manual reference; last-axis default unchanged.
5. transpose-axes-backward -- general-permutation transpose_axes on the
   tape (current tape transpose is reverse-axes only). RS3. TDD: gradcheck
   through a non-trivial permutation.
6. bf16-f16-reinterpret -- bf16/f16 decode in reinterpret for real model
   weights. RS6. TDD: known bit patterns decode to expected f64.
7. relay-and-close -- mark RS1-RS6 resolved in docs/sw-mlpl-findings.md,
   refresh CHANGES + wiki errata, mark the saga shipped, rebuild binaries
   (repl/build release+debug, serve release). --done.
