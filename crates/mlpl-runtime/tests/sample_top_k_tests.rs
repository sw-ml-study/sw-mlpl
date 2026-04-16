//! Tests for the categorical `sample(logits, temperature, seed)` and
//! `top_k(logits, k)` built-ins introduced in saga 13 step 005.

use mlpl_array::DenseArray;
use mlpl_runtime::call_builtin;

fn scalar(x: f64) -> DenseArray {
    DenseArray::from_scalar(x)
}

fn vec1(xs: &[f64]) -> DenseArray {
    DenseArray::from_vec(xs.to_vec())
}

fn argmax(xs: &[f64]) -> usize {
    let mut best = 0usize;
    let mut bv = xs[0];
    for (i, &v) in xs.iter().enumerate().skip(1) {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

#[test]
fn sample_is_deterministic_for_same_seed() {
    let logits = vec1(&[0.1, 0.5, -0.2, 1.3, 0.0]);
    let a = call_builtin("sample", vec![logits.clone(), scalar(1.0), scalar(42.0)]).unwrap();
    let b = call_builtin("sample", vec![logits, scalar(1.0), scalar(42.0)]).unwrap();
    assert_eq!(a.shape().dims(), &[] as &[usize]);
    assert_eq!(a.data(), b.data());
}

#[test]
fn sample_differs_for_different_seed_in_aggregate() {
    let logits = vec1(&[0.1, 0.2, 0.3, 0.4, 0.5]);
    let mut seen = std::collections::BTreeSet::new();
    for s in 0..50u64 {
        let r = call_builtin(
            "sample",
            vec![logits.clone(), scalar(1.0), scalar(s as f64)],
        )
        .unwrap();
        seen.insert(r.data()[0] as usize);
    }
    // High-entropy logits with 50 different seeds should hit at least
    // 3 of the 5 classes; if every seed gave the same id that would be
    // a very strong sign the seed is not threaded through.
    assert!(seen.len() >= 3, "only saw {} distinct ids", seen.len());
}

#[test]
fn sample_temperature_zero_matches_argmax() {
    let cases = [
        vec![0.1, 0.5, -0.2, 1.3, 0.0],
        vec![-1.0, -2.0, -0.5, -0.4, -3.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
        vec![0.0; 4],
    ];
    for c in &cases {
        for seed in [0u64, 1, 7, 99] {
            let r =
                call_builtin("sample", vec![vec1(c), scalar(0.0), scalar(seed as f64)]).unwrap();
            assert_eq!(r.data()[0] as usize, argmax(c), "case={:?}", c);
        }
    }
}

#[test]
fn top_k_keeps_top_k_finite_rest_neg_inf() {
    let logits = vec1(&[0.1, 3.0, -0.2, 2.5, 1.0]);
    let out = call_builtin("top_k", vec![logits, scalar(2.0)]).unwrap();
    let d = out.data();
    assert_eq!(out.shape().dims(), &[5]);
    // Top 2 values are 3.0 (idx 1) and 2.5 (idx 3). They must remain
    // finite and equal to the originals; everything else must be -inf.
    assert_eq!(d[1], 3.0);
    assert_eq!(d[3], 2.5);
    assert!(d[0].is_infinite() && d[0].is_sign_negative());
    assert!(d[2].is_infinite() && d[2].is_sign_negative());
    assert!(d[4].is_infinite() && d[4].is_sign_negative());
}

#[test]
fn top_k_full_v_is_identity() {
    let logits_data = vec![0.1, 3.0, -0.2, 2.5, 1.0];
    let out = call_builtin("top_k", vec![vec1(&logits_data), scalar(5.0)]).unwrap();
    assert_eq!(out.data(), logits_data.as_slice());
}

#[test]
fn top_k_one_then_sample_is_argmax_at_any_temperature() {
    let logits = vec![0.1, 3.0, -0.2, 2.5, 1.0];
    let masked = call_builtin("top_k", vec![vec1(&logits), scalar(1.0)]).unwrap();
    for &t in &[0.0_f64, 0.5, 1.0, 4.0, 100.0] {
        for seed in [0u64, 1, 7, 99] {
            let r = call_builtin(
                "sample",
                vec![masked.clone(), scalar(t), scalar(seed as f64)],
            )
            .unwrap();
            assert_eq!(
                r.data()[0] as usize,
                argmax(&logits),
                "temperature={t} seed={seed}"
            );
        }
    }
}

#[test]
fn sample_high_temperature_distribution_matches_softmax() {
    // At very high temperature the distribution flattens toward
    // uniform. Use moderate temperature so we get a meaningful but not
    // trivial target distribution. logits=[1,2,3] → softmax target.
    let logits = vec![1.0_f64, 2.0, 3.0];
    let temperature = 2.0_f64;
    let scaled: Vec<f64> = logits.iter().map(|x| x / temperature).collect();
    let m = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|x| (x - m).exp()).collect();
    let z: f64 = exps.iter().sum();
    let p: Vec<f64> = exps.iter().map(|e| e / z).collect();

    let n: usize = 10000;
    let mut counts = vec![0u64; logits.len()];
    for s in 0..n {
        let r = call_builtin(
            "sample",
            vec![vec1(&logits), scalar(temperature), scalar(s as f64)],
        )
        .unwrap();
        let id = r.data()[0] as usize;
        assert!(id < logits.len());
        counts[id] += 1;
    }
    // Pearson chi-square goodness-of-fit. With 3 categories => df=2.
    // Critical value at p=0.001 is ~13.82. Use a generous bound.
    let mut chi2 = 0.0_f64;
    for (c, pi) in counts.iter().zip(p.iter()) {
        let expected = (n as f64) * pi;
        let diff = *c as f64 - expected;
        chi2 += diff * diff / expected;
    }
    assert!(
        chi2 < 13.82,
        "chi2={chi2} counts={counts:?} expected_p={p:?}"
    );
}

#[test]
fn sample_rejects_non_vector_logits() {
    let logits =
        DenseArray::new(mlpl_array::Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let err = call_builtin("sample", vec![logits, scalar(1.0), scalar(0.0)]).unwrap_err();
    assert!(format!("{err}").contains("rank"), "err={err}");
}

#[test]
fn sample_rejects_non_scalar_temperature() {
    let logits = vec1(&[0.1, 0.2, 0.3]);
    let err = call_builtin("sample", vec![logits, vec1(&[1.0, 1.0]), scalar(0.0)]).unwrap_err();
    assert!(
        format!("{err}").contains("temperature") || format!("{err}").contains("scalar"),
        "err={err}"
    );
}

#[test]
fn sample_rejects_negative_temperature() {
    let logits = vec1(&[0.1, 0.2, 0.3]);
    let err = call_builtin("sample", vec![logits, scalar(-0.5), scalar(0.0)]).unwrap_err();
    assert!(format!("{err}").contains("temperature"), "err={err}");
}

#[test]
fn top_k_rejects_non_vector() {
    let logits =
        DenseArray::new(mlpl_array::Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let err = call_builtin("top_k", vec![logits, scalar(1.0)]).unwrap_err();
    assert!(format!("{err}").contains("rank"), "err={err}");
}

#[test]
fn top_k_rejects_zero_or_oversize_k() {
    let logits = vec1(&[0.1, 0.2, 0.3]);
    let err0 = call_builtin("top_k", vec![logits.clone(), scalar(0.0)]).unwrap_err();
    assert!(format!("{err0}").contains("k"), "err={err0}");
    let err_big = call_builtin("top_k", vec![logits, scalar(99.0)]).unwrap_err();
    assert!(format!("{err_big}").contains("k"), "err={err_big}");
}
