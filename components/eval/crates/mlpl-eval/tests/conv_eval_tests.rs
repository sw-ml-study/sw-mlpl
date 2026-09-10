//! Convolution capability pinned as a regression (demo-ml-utils). The
//! weighted, shared-kernel convolution is expressible on the current
//! interpreter TODAY -- via `windows` + `reshape` + `matmul` -- with NO
//! rank-broadcasting. Trailing-axis rank broadcast (the "C2" ergonomics)
//! only adds the prettier elementwise `reduce(:add, kernel * windows(..))`
//! spelling; this file guards the underlying capability so it cannot
//! silently regress while that spelling is still pending.

use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    eval_program(&stmts, &mut Environment::new())
}

#[test]
fn weighted_conv_via_matmul_needs_no_rank_broadcast() {
    // img [C=2, H=3, W=3]; kernel [C, kh, kw] flattened to [C*kh*kw].
    // windows(img,[2,2]) is [out_y, out_x, C, kh, kw] = [2,2,2,2,2]; flatten
    // to [out_y*out_x, C*kh*kw] = [4, 8] and matmul the flattened kernel.
    let src = "img = reshape(range(18), [2, 3, 3])\n\
               kernel = range(8)\n\
               pf = reshape(windows(img, [2, 2]), [4, 8])\n\
               matmul(pf, kernel)";
    let y = eval(src).unwrap();
    assert_eq!(y.shape(), &Shape::new(vec![4]));
    // position (0,0): ch0 [0,1,3,4].[0,1,2,3] + ch1 [9,10,12,13].[4,5,6,7]
    //   = (0+1+6+12) + (36+50+72+91) = 19 + 249 = 268
    assert_eq!(y.data(), &[268.0, 296.0, 352.0, 380.0]);
}

#[test]
fn windows_axis_order_is_positions_then_channel_then_window() {
    // The load-bearing convolution layout: [out_y, out_x, C, kh, kw], so a
    // flattened kernel [C*kh*kw] lines up with each patch row after reshape.
    let a = eval("shape(windows(reshape(range(18), [2, 3, 3]), [2, 2]))").unwrap();
    assert_eq!(a.data(), &[2.0, 2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn box_filter_conv_is_reduce_over_the_receptive_field() {
    // An all-ones kernel needs no kernel term at all: reduce the trailing
    // receptive-field axes (C, kh, kw = axes 4, 3, 2) of the patch stack.
    let src = "img = reshape(range(18), [2, 3, 3])\n\
               reduce_add(reduce_add(reduce_add(windows(img, [2, 2]), 4), 3), 2)";
    let y = eval(src).unwrap();
    assert_eq!(y.shape(), &Shape::new(vec![2, 2]));
    // per position, sum of both channels' 2x2 patch: 52, 60, 76, 84
    assert_eq!(y.data(), &[52.0, 60.0, 76.0, 84.0]);
}

#[test]
fn the_elementwise_kernel_spelling_matches_the_matmul_conv() {
    // C2 (rank broadcasting) landed: a rank-3 kernel [C,kh,kw] now
    // broadcasts against rank-5 patches [oy,ox,C,kh,kw], so the
    // elementwise spelling computes the SAME convolution as the im2col
    // matmul form above -- pinning the requirement (equal results), not
    // merely that the multiply no longer errors.
    let src = "img = reshape(range(18), [2, 3, 3])\n\
               k = reshape(range(8), [2, 2, 2])\n\
               reduce(:add, windows(img, [2, 2]) * k, [2, 3, 4])";
    let y = eval(src).unwrap();
    assert_eq!(y.shape(), &Shape::new(vec![2, 2]));
    assert_eq!(y.data(), &[268.0, 296.0, 352.0, 380.0]);
}
