Create docs/syntax-core-v1.md -- a small syntax design spike for MLPL v1.

This must exist BEFORE any parser implementation begins. Keep it concise (under 150 lines). Cover:

1. Numeric literals:
   - Integers: 1, 42, -3
   - Floats: 1.5, -0.25, 3.14

2. Identifiers:
   - Lowercase letters and underscores: x, my_var, result
   - Start with letter, contain letters/digits/underscores

3. Array literals:
   - Vector: [1, 2, 3]
   - Matrix: [[1, 2], [3, 4]]
   - Nested: [[1, 2, 3], [4, 5, 6]]

4. Function call form:
   - Named functions: add(1, 2), reshape([1,2,3,4], [2,2])
   - Built-in primitives for v1: add, sub, mul, div, reshape, transpose, shape, iota, reduce_add

5. Assignment:
   - x = expr

6. Comments:
   - Line comments: # this is a comment

7. Delimiters and whitespace:
   - Comma-separated in arrays and function args
   - Newline or semicolon separates statements
   - Whitespace is not significant (except as separator)

8. Examples section showing 5-10 sample expressions

Design principle: keep it boring and easy. Do NOT try to replicate APL/BQN symbolic notation yet. That comes later as Unicode aliases. The v1 syntax should look like a simple scripting language that happens to have array operations.

Reference docs/clarifications.txt decision #8 for context.