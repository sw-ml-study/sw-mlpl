//! `windows` at the interpreter level (demo-ml-utils C1): the general
//! overlapping sliding-window rearrangement, exercised the way the
//! moving-average and convolution demos use it.

use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, &mut Environment::new())
}

#[test]
fn the_probe_shape_matches() {
    // demo-ml-utils probes/sliding-windows.mlpl
    let a = eval("windows(reshape(range(16), [4, 4]), [3, 3])").unwrap();
    assert_eq!(a.shape(), &Shape::new(vec![2, 2, 3, 3]));
}

#[test]
fn moving_average_is_a_reduce_over_windows() {
    // The direct J-style form the moving-average demo could not write
    // before: reduce over the window axis, divide by the width.
    let a = eval("reduce_add(windows([10.0, 20.0, 30.0, 40.0], [3]), 1) / 3").unwrap();
    assert_eq!(a.data(), &[20.0, 30.0]);
}

#[test]
fn two_d_patch_sums_reduce_over_the_window_axes() {
    // Sum each 2x2 patch of a 3x3 grid (an all-ones-kernel convolution):
    // reduce over the two window axes (3 and 4). The full `* kernel` form
    // needs rank broadcasting (C2, a later phase); this tests windows +
    // reduce, which stand alone.
    // windows(img,[2,2]) is [out_y, out_x, kh, kw] = axes [0,1,2,3];
    // reduce the two window axes 3 then 2.
    let src = "img = reshape(range(9), [3, 3])\n\
               reduce_add(reduce_add(windows(img, [2, 2]), 3), 2)";
    let y = eval(src).unwrap();
    assert_eq!(y.shape(), &Shape::new(vec![2, 2]));
    // patches: [[0,1,3,4],[1,2,4,5],[3,4,6,7],[4,5,7,8]] -> sums 8,12,20,24
    assert_eq!(y.data(), &[8.0, 12.0, 20.0, 24.0]);
}

#[test]
fn strided_windows_skip_positions() {
    let a = eval("shape(windows(range(6), [2], [2]))").unwrap();
    assert_eq!(a.data(), &[3.0, 2.0]);
}

#[test]
fn a_window_larger_than_the_axis_errors() {
    assert!(eval("windows([1.0, 2.0, 3.0], [4])").is_err());
}
