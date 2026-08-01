//! Resident-handle parity: every `DeviceOps` op agrees with f64 host
//! math within the fp32 tolerance, chains stay LAZY (one download
//! at the end), and foreign/CPU handles fail loudly. Triple-gated
//! like every MLX suite.

#![cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]

use mlpl_array::{DenseArray, Shape};
use mlpl_mlx_handle::register_mlx_device_ops;
use mlpl_tensor_handle::{AxisKind, BinKind, TensorHandle, UnaryKind, upload};

const FP32_TOL: f64 = 1e-5;

fn setup(dims: Vec<usize>, data: Vec<f64>) -> TensorHandle {
    register_mlx_device_ops();
    upload(&DenseArray::new(Shape::new(dims), data).unwrap()).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < FP32_TOL, "[{i}]: {a} vs {e}");
    }
}

#[test]
fn binaries_match_host_math() {
    let a = setup(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = setup(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let cases: Vec<(BinKind, Vec<f64>)> = vec![
        (BinKind::Add, vec![6.0, 8.0, 10.0, 12.0]),
        (BinKind::Sub, vec![-4.0, -4.0, -4.0, -4.0]),
        (BinKind::Mul, vec![5.0, 12.0, 21.0, 32.0]),
        (BinKind::Div, vec![0.2, 2.0 / 6.0, 3.0 / 7.0, 0.5]),
        (BinKind::Matmul, vec![19.0, 22.0, 43.0, 50.0]),
    ];
    for (op, expected) in cases {
        let out = a.dev_binary(op, &b).unwrap();
        assert!(out.is_dev(), "{op:?} result stays resident");
        assert_close(out.to_dense().data(), &expected);
    }
}

#[test]
fn unaries_match_host_math() {
    let vals = [0.5, -1.25, 2.0, -0.1];
    let a = setup(vec![2, 2], vals.to_vec());
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let cases: Vec<(UnaryKind, Vec<f64>)> = vec![
        (UnaryKind::Neg, vals.iter().map(|x| -x).collect()),
        (UnaryKind::Exp, vals.iter().map(|x| x.exp()).collect()),
        (UnaryKind::Tanh, vals.iter().map(|x| x.tanh()).collect()),
        (UnaryKind::Sigmoid, vals.iter().map(|&x| sig(x)).collect()),
        (UnaryKind::Relu, vals.iter().map(|x| x.max(0.0)).collect()),
        (UnaryKind::Transpose, vec![0.5, 2.0, -1.25, -0.1]),
    ];
    for (op, expected) in cases {
        assert_close(a.dev_unary(op).unwrap().to_dense().data(), &expected);
    }
    // Log on positive inputs.
    let p = setup(vec![3], vec![0.5, 1.0, 4.0]);
    let expected: Vec<f64> = [0.5f64, 1.0, 4.0].iter().map(|x| x.ln()).collect();
    assert_close(
        p.dev_unary(UnaryKind::Log).unwrap().to_dense().data(),
        &expected,
    );
}

#[test]
fn axis_ops_match_host_math() {
    let a = setup(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // Row softmax of [1,2,3]: exp shifted by max, normalized.
    let row = |v: [f64; 3]| {
        let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = v.iter().map(|x| (x - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.into_iter().map(|x| x / s).collect::<Vec<_>>()
    };
    let mut expected = row([1.0, 2.0, 3.0]);
    expected.extend(row([4.0, 5.0, 6.0]));
    let sm = a.dev_axis(AxisKind::Softmax, Some(1), false).unwrap();
    assert_close(sm.to_dense().data(), &expected);
    // Sum along axis 0, mean over all.
    let s0 = a.dev_axis(AxisKind::Sum, Some(0), false).unwrap();
    assert_eq!(s0.dims(), vec![3]);
    assert_close(s0.to_dense().data(), &[5.0, 7.0, 9.0]);
    let s0k = a.dev_axis(AxisKind::Sum, Some(0), true).unwrap();
    assert_eq!(s0k.dims(), vec![1, 3], "keep_dims preserves rank");
    let m = a.dev_axis(AxisKind::Mean, None, false).unwrap();
    assert_close(m.to_dense().data(), &[3.5]);
}

#[test]
fn cross_entropy_matches_the_cpu_formula() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let a = setup(vec![2, 4], logits.clone());
    let targets = [2usize, 0];
    let mut total = 0.0;
    for (i, &t) in targets.iter().enumerate() {
        let row = &logits[i * 4..(i + 1) * 4];
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + row.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
        total += lse - row[t];
    }
    let out = a.dev_cross_entropy(&targets).unwrap();
    assert_close(out.to_dense().data(), &[total / 2.0]);
}

#[test]
fn chained_ops_stay_lazy_with_one_download() {
    // (a @ b + a).tanh() summed -- five graph nodes, ZERO evals
    // until the single to_dense at the end.
    let a = setup(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]);
    let b = setup(vec![2, 2], vec![0.5, 0.6, 0.7, 0.8]);
    let out = a
        .dev_binary(BinKind::Matmul, &b)
        .unwrap()
        .dev_binary(BinKind::Add, &a)
        .unwrap()
        .dev_unary(UnaryKind::Tanh)
        .unwrap()
        .dev_axis(AxisKind::Sum, None, false)
        .unwrap();
    assert!(out.is_dev());
    // Host reference.
    let (av, bv): ([f64; 4], [f64; 4]) = ([0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]);
    let mm = [
        av[0] * bv[0] + av[1] * bv[2],
        av[0] * bv[1] + av[1] * bv[3],
        av[2] * bv[0] + av[3] * bv[2],
        av[2] * bv[1] + av[3] * bv[3],
    ];
    let expected: f64 = mm.iter().zip(&av).map(|(m, x)| (m + x).tanh()).sum();
    assert_close(out.to_dense().data(), &[expected]);
}

#[test]
fn mixed_cpu_dev_binary_uploads_the_host_side() {
    let a = setup(vec![2], vec![1.0, 2.0]);
    let c = TensorHandle::from(DenseArray::new(Shape::new(vec![2]), vec![10.0, 20.0]).unwrap());
    let out = a.dev_binary(BinKind::Add, &c).unwrap();
    assert!(out.is_dev());
    assert_close(out.to_dense().data(), &[11.0, 22.0]);
}

#[test]
fn reshape_roundtrips() {
    let a = setup(vec![2, 3], (0..6).map(f64::from).collect());
    let r = a.dev_reshape(&[3, 2]).unwrap();
    assert_eq!(r.dims(), vec![3, 2]);
    assert_close(r.to_dense().data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}
