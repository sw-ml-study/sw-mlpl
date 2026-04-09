# Appendix: Python ML Library Comparison

MLPL is inspired by the great Python machine learning libraries, but it takes a different approach to syntax and safety.

| Feature | NumPy / PyTorch | MLPL |
|---------|-----------------|------|
| **Loops** | `for` loops (slow) | Implicit (vectorized) |
| **Broadcasting** | Implicit / Rules-based | Explicit / Named Axes |
| **Gradients** | `backward()` | Automated Trace |
| **Visualization** | Matplotlib (External) | `viz()` (Native SVG) |

## From NumPy to MLPL

If you are used to writing:
```python
x = np.array([1, 2, 3])
y = x * 10
```

In MLPL, it is nearly identical:
```
x = [1, 2, 3]
y = x * 10
```

The key difference is in **dimensionality**. While NumPy uses integer indices (axis 0, axis 1), MLPL encourages you to use **Named Axes**, making your code self-documenting and preventing shape-mismatch bugs.
