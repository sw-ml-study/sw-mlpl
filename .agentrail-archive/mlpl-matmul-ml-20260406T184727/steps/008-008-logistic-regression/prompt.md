Create a logistic regression training demo.

Write demos/logistic_regression.mlpl that trains a simple model:

1. Dataset: AND gate (or simple 2D linearly separable data)
   - X = [[0,0],[0,1],[1,0],[1,1]] (4 samples, 2 features)
   - y = [0, 0, 0, 1] (AND gate labels)

2. Model: logistic regression
   - w = initial weights (zeros or small values)
   - b = initial bias (0)
   - Forward: z = matmul(X, w) + b; pred = sigmoid(z)
   - Loss: binary cross-entropy (or MSE for simplicity)
   - Gradient: manual computation
     - dz = pred - y
     - dw = matmul(transpose(X), dz) / n
     - db = mean(dz)
   - Update: w = w - lr * dw; b = b - lr * db

3. Training: use eval_n or explicit repeated lines for ~100 steps

4. Show: initial loss, final loss, predictions, accuracy

The demo should converge (loss decreases, accuracy approaches 1.0
or at least significantly improves).

TDD:
- Run the demo script via -f flag
- Verify it produces output without errors
- Verify final predictions are close to expected labels

Allowed: demos/, crates/mlpl-eval/
