Add a toy attention-pattern demo.

1. demos/attention.mlpl: build a small query matrix Q (e.g. 6x4) and key matrix K (6x4) of randn values. Compute the attention scores matmul(Q, transpose(K)) / sqrt(d), apply softmax along axis 1, and render the resulting 6x6 attention matrix as a heatmap via svg(scores, "heatmap"). No real model -- just the dot-product attention pattern.
2. Optional: also render the raw scores (pre-softmax) as a second heatmap so the user can see the effect of softmax row-normalization.
3. Add the demo to the web dropdown and a tutorial lesson "Attention Patterns".
4. Depends on softmax built-in from step 005 and randn from step 001.
5. Rebuild pages.
6. Allowed: demos/, apps/mlpl-web/src/, docs/