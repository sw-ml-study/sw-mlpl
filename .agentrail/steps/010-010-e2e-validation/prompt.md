End-to-end validation: run all syntax-core-v1.md examples through the REPL and verify correct output.

Write an integration test (or script) that feeds each example from docs/syntax-core-v1.md into the parse+eval pipeline and asserts the expected output:

1. "1 + 2" -> 3
2. "[1, 2, 3] + [4, 5, 6]" -> 5 7 9
3. "[1, 2, 3] * 10" -> 10 20 30
4. "x = iota(12)\nm = reshape(x, [3, 4])" -> 3x4 matrix
5. "t = transpose(m)" -> 4x3 matrix
6. "shape(t)" -> [4, 3]
7. "reduce_add([1, 2, 3, 4, 5])" -> 15
8. Multi-step computation:
   data = [1, 2, 3, 4, 5, 6]
   grid = reshape(data, [2, 3])
   scaled = grid * 2
   result = reduce_add(scaled)
   -> 42

Fix any bugs found. This is the acceptance test for the saga.

Allowed: all crates and apps
