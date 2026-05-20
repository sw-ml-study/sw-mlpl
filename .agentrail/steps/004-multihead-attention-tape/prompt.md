Tier 1 saga step 004: lower multi-head attention onto the same tape primitives that single-head already uses, so multi-head ViTs actually train end-to-end.

Today crates/mlpl-eval/src/grad.rs has full forward + backward for attention(d_model, h=1, seed); for h>1 the tape stops at the per-head split and the per-head Q/K/V projections never see gradients. Visible symptom: vit_multihead_quick.mlpl loss drops a few hundredths over 30 steps and accuracy hovers around 0.5.

Lower multi-head onto the existing tape primitives. The per-head forward is identical to single-head:
  Q_h = X @ Wq_h          # one of h slabs
  K_h = X @ Wk_h
  V_h = X @ Wv_h
  scores_h = (Q_h @ K_h^T) / sqrt(d_h)
  weights_h = softmax(scores_h, axis=-1)
  out_h = weights_h @ V_h
The multi-head wrapper is then:
  out = stack([out_0, ..., out_{h-1}], axis=-1)  # along d_model axis
  Y = out @ Wo

The Stack tape op (saga 29 step 008) is already in place; use it. Use the now-general concat backward (step 002) where needed.

TDD:
- RED: finite-difference gradcheck on attention(d=8, h=2, seed=0) with input [T=4, d=8]. Compare analytic gradient against numerical gradient elementwise, tolerance 1e-3 absolute.
- GREEN: implement the multi-head lowering in grad.rs; dispatch on h>1 to the new path. The h=1 fast path stays untouched.
- Integration test: in crates/mlpl-eval/tests/, add an integration that runs the equivalent of vit_multihead_quick.mlpl's training block (or a smaller version: 4 samples, 50 adam steps, attention(d=8, h=2)) and asserts loss decreases by at least 30% from step 0 to step 50.

Quality gates: cargo test -p mlpl-eval (release for the integration test); cargo clippy -p mlpl-eval --all-targets --all-features -- -D warnings; cargo fmt; sw-checklist hold-or-lower.

This is the headline step of the saga -- after it lands the user-facing demos behave dramatically differently. Step 005 updates the demo intros/takeaways to reflect that.