Build an interactive tutorial that teaches MLPL step by step.

The tutorial is a guided walkthrough built into the web UI.
Users progress through lessons, each introducing new concepts
with explanations and try-it-yourself prompts.

1. Add a "Tutorial" button/tab to the web UI that switches
   from free REPL mode to guided tutorial mode.

2. Tutorial structure (progressive difficulty):

   Lesson 1: Hello Numbers
   - Scalar literals and arithmetic: 1 + 2, 3 * 4, 10 / 3
   - Order of operations: 2 + 3 * 4, (2 + 3) * 4
   - Negative numbers: -5, -3 + 7

   Lesson 2: Arrays
   - Array literals: [1, 2, 3]
   - Element-wise arithmetic: [1,2,3] + [4,5,6], [1,2,3] * 10
   - Scalar broadcasting explained

   Lesson 3: Variables
   - Assignment: x = 42, name = [1,2,3]
   - Using variables in expressions: x + 1, x * x
   - Reassignment

   Lesson 4: Built-in Functions
   - iota(n): generating sequences
   - shape(a), rank(a): inspecting arrays
   - reduce_add, reduce_mul: summarizing data

   Lesson 5: Matrices
   - reshape: turning vectors into matrices
   - transpose: flipping rows and columns
   - Matrix display and indexing concepts
   - Axis-specific reductions

   Lesson 6: Linear Algebra
   - dot product: dot(a, b)
   - Matrix multiplication: matmul(A, B)
   - Practical example: transforming data

   Lesson 7: Math and Activations
   - exp, log, sqrt, abs
   - sigmoid, tanh_fn for ML
   - pow for element-wise power

   Lesson 8: Comparisons and Logic
   - gt, lt, eq returning 0/1 arrays
   - Using comparisons for filtering/masking
   - mean for statistics

   Lesson 9: Loops and Iteration
   - repeat N { body }
   - Accumulation patterns
   - Building toward gradient descent

   Lesson 10: Machine Learning -- Logistic Regression
   - Setting up a dataset (AND gate)
   - Initializing weights: zeros, ones
   - Forward pass: matmul + sigmoid
   - Computing loss and gradients
   - Training loop with gradient descent
   - Measuring accuracy
   - Full working model

3. Each lesson has:
   - Title and brief explanation text
   - Example code shown (user can run it or type their own)
   - "Try it" prompts encouraging experimentation
   - "Next" button to advance
   - Progress indicator (Lesson X of 10)

4. Tutorial state persists in the REPL environment so
   variables from earlier lessons are available later.

5. Tutorial content is data-driven (array of lesson structs)
   so it is easy to add more lessons later.

Allowed: apps/mlpl-web/