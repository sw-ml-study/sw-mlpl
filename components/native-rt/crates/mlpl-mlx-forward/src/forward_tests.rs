//! Parity tests: each MLX forward primitive vs a hand-computed f32
//! reference. These build forward VALUES only (no value_and_grad), so
//! they are safe to run in parallel.

use crate::{causal_attention, causal_mask, cross_entropy, embed, rms_norm};
use mlx_rs::Array;

#[test]
fn embed_selects_table_rows_via_onehot() {
    let _mlx = crate::MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // 2 tokens over V=3, d=2. Tokens 1 and 2 select table rows 1, 2.
    let onehot = Array::from_slice(&[0.0f32, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]);
    let table = Array::from_slice(&[10.0f32, 11.0, 20.0, 21.0, 30.0, 31.0], &[3, 2]);
    let out = embed(&onehot, &table).unwrap();
    assert_eq!(out.as_slice::<f32>(), &[20.0, 21.0, 30.0, 31.0]);
}

#[test]
fn rms_norm_matches_reference() {
    let _mlx = crate::MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // x = [3, 4], gamma = [1, 1], eps = 0. mean(x^2) = 12.5,
    // 1/sqrt = 0.282843 -> [0.848528, 1.131371].
    let x = Array::from_slice(&[3.0f32, 4.0], &[1, 2]);
    let gamma = Array::from_slice(&[1.0f32, 1.0], &[2]);
    let out = rms_norm(&x, &gamma, 0.0).unwrap();
    let o = out.as_slice::<f32>();
    assert!((o[0] - 0.848528).abs() < 1e-4, "{}", o[0]);
    assert!((o[1] - 1.131371).abs() < 1e-4, "{}", o[1]);
}

#[test]
fn cross_entropy_matches_reference() {
    let _mlx = crate::MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // logits = [1, 2, 3], target = 2. logsumexp = 3.40760, picked = 3,
    // ce = 0.40760.
    let logits = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]);
    let onehot = Array::from_slice(&[0.0f32, 0.0, 1.0], &[1, 3]);
    let ce = cross_entropy(&logits, &onehot).unwrap();
    assert!(
        (ce.item::<f32>() - 0.40760).abs() < 1e-4,
        "{}",
        ce.item::<f32>()
    );
}

#[test]
fn causal_attention_matches_reference() {
    let _mlx = crate::MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // T=2, d=2, identity Q/K/V/O, x = I. Row 0 can only attend to key 0
    // (causal mask) -> out0 = v0 = [1, 0]. Row 1 attends both: scores
    // [0, 1/sqrt2], softmax -> [0.3302, 0.6698], out1 = that mix of v.
    let ident = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
    let x = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
    let out = causal_attention(&x, &ident, &ident, &ident, &ident, &causal_mask(2)).unwrap();
    let o = out.as_slice::<f32>();
    let want = [1.0f32, 0.0, 0.3302, 0.6698];
    for (got, w) in o.iter().zip(want) {
        assert!((got - w).abs() < 2e-3, "attention out {o:?} vs {want:?}");
    }
}
