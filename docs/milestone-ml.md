# ML Foundations Milestone (v0.2.0)

Building on the MVP, this milestone adds matrix multiplication, math
functions, and enough infrastructure to train a logistic regression model.

## Delivered

- [x] Vector dot product (`dot`)
- [x] Matrix multiplication (`matmul`)
- [x] Element-wise math functions (`exp`, `log`, `sqrt`, `abs`)
- [x] Power function (`pow`)
- [x] ML activations (`sigmoid`, `tanh_fn`)
- [x] Comparison operators (`gt`, `lt`, `eq`)
- [x] Statistical functions (`mean`)
- [x] Array constructors (`zeros`, `ones`, `fill`)
- [x] Loop construct (`repeat N { ... }`)
- [x] Logistic regression training demo (AND gate)
- [x] Updated REPL `:help` with all 23 built-in functions
- [x] Updated README with full function reference

## Built-in Function Count

- MVP (v0.1): 9 functions
- ML milestone (v0.2): 23 functions (+14)

## Demo: Logistic Regression

`demos/logistic_regression.mlpl` trains a 2-parameter model to learn the
AND gate using 300 steps of gradient descent. The model converges to 100%
accuracy on the 4-sample dataset.

```bash
cargo run -p mlpl-repl -- -f demos/logistic_regression.mlpl
```

## What's Next

- [ ] Rank/cell semantics
- [ ] Multi-layer neural network demo
- [ ] Browser-based visual trace viewer
- [ ] Random array constructors
