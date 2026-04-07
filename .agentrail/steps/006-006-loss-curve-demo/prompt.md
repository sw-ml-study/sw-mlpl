Update the logistic regression demo and tutorial to show a
loss curve via svg().

1. Logistic regression demo (demos/logistic_regression.mlpl):
   - During training, append the loss to a vector each iteration
   - After training, render svg(losses, "line")
   - Also render svg(predictions, "bar") to compare to expected

2. Add a new demo: demos/loss_curve.mlpl
   - Simple 1D linear regression with svg loss-curve viz
   - Demonstrates the pattern in isolation

3. Update web demo selector (apps/mlpl-web/src/demos.rs):
   - Add "Loss Curve" demo
   - Update Logistic Regression demo to include svg() lines

4. Update interactive tutorial:
   - Add Lesson 11 "Visualizing Data": svg() with scatter,
     line, bar, heatmap examples
   - Update Lesson 10 (Logistic Regression) to include the
     loss curve visualization

5. Quality gates pass, pages rebuilt

Allowed: demos/, apps/mlpl-web/src/, docs/