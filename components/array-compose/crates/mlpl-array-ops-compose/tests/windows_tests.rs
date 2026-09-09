//! `windows` forward: stride-configurable sliding-window extraction, the
//! primitive under a moving average (1-D) and a convolution patch stack
//! (2-D). demo-ml-utils C1.

use mlpl_array::{DenseArray, Shape};
use mlpl_array_ops_compose::WindowsExt;

fn arr(dims: &[usize], data: &[f64]) -> DenseArray {
    DenseArray::new(Shape::new(dims.to_vec()), data.to_vec()).unwrap()
}

#[test]
fn one_d_windows_are_the_moving_average_windows() {
    // [10,20,30,40] windows of 3, stride 1 -> [[10,20,30],[20,30,40]].
    let w = arr(&[4], &[10.0, 20.0, 30.0, 40.0])
        .windows(&[3], &[1])
        .unwrap();
    assert_eq!(w.shape().dims(), &[2, 3]);
    assert_eq!(w.data(), &[10.0, 20.0, 30.0, 20.0, 30.0, 40.0]);
}

#[test]
fn two_d_windows_are_convolution_patches() {
    // 3x3 grid 0..8, 2x2 windows stride 1 -> [2,2,2,2].
    let g = arr(&[3, 3], &(0..9).map(|i| i as f64).collect::<Vec<_>>());
    let w = g.windows(&[2, 2], &[1, 1]).unwrap();
    assert_eq!(w.shape().dims(), &[2, 2, 2, 2]);
    assert_eq!(&w.data()[0..4], &[0.0, 1.0, 3.0, 4.0]); // patch (0,0)
    assert_eq!(&w.data()[12..16], &[4.0, 5.0, 7.0, 8.0]); // patch (1,1)
}

#[test]
fn positions_lead_and_non_windowed_axes_trail_next_to_the_window() {
    // [C=2, W=4] windows of 3 over the last axis -> [out=2, C=2, win=3].
    // Positions lead; the non-windowed C sits with the window axis, so
    // the layout at each position is [ch0 window, ch1 window].
    let x = arr(&[2, 4], &[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]);
    let w = x.windows(&[3], &[1]).unwrap();
    assert_eq!(w.shape().dims(), &[2, 2, 3]);
    // position 0: ch0 [0,1,2], ch1 [10,11,12]
    assert_eq!(&w.data()[0..6], &[0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
    // position 1: ch0 [1,2,3], ch1 [11,12,13]
    assert_eq!(&w.data()[6..12], &[1.0, 2.0, 3.0, 11.0, 12.0, 13.0]);
}

#[test]
fn multi_channel_2d_lays_out_for_kernel_alignment() {
    // The load-bearing convolution ordering (demo-ml-utils reference):
    // [C, H, W].windows([kh, kw]) -> [out_y, out_x, C, kh, kw], so a
    // kernel [C, kh, kw] aligns by trailing position with no transpose.
    let x = arr(&[2, 3, 3], &(0..18).map(|i| i as f64).collect::<Vec<_>>());
    let w = x.windows(&[2, 2], &[1, 1]).unwrap();
    assert_eq!(w.shape().dims(), &[2, 2, 2, 2, 2]);
    // At output position (0,0): channel 0's 2x2 patch, then channel 1's.
    assert_eq!(&w.data()[0..4], &[0.0, 1.0, 3.0, 4.0]); // ch0 patch (0,0)
    assert_eq!(&w.data()[4..8], &[9.0, 10.0, 12.0, 13.0]); // ch1 patch (0,0)
}

#[test]
fn non_unit_stride_skips_positions() {
    // [0..6) windows of 2, stride 2 -> non-overlapping [[0,1],[2,3],[4,5]].
    let w = arr(&[6], &(0..6).map(|i| i as f64).collect::<Vec<_>>())
        .windows(&[2], &[2])
        .unwrap();
    assert_eq!(w.shape().dims(), &[3, 2]);
    assert_eq!(w.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    // stride 3 over length 7, window 2 -> positions 0 and 3 -> [[0,1],[3,4]].
    let w2 = arr(&[7], &(0..7).map(|i| i as f64).collect::<Vec<_>>())
        .windows(&[2], &[3])
        .unwrap();
    assert_eq!(w2.shape().dims(), &[2, 2]);
    assert_eq!(w2.data(), &[0.0, 1.0, 3.0, 4.0]);
}

#[test]
fn a_window_equal_to_the_axis_yields_one_position() {
    let w = arr(&[3], &[1.0, 2.0, 3.0]).windows(&[3], &[1]).unwrap();
    assert_eq!(w.shape().dims(), &[1, 3]);
    assert_eq!(w.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn errors_are_clean() {
    let a = arr(&[3], &[1.0, 2.0, 3.0]);
    assert!(a.windows(&[4], &[1]).is_err()); // window larger than axis
    assert!(a.windows(&[2, 2], &[1, 1]).is_err()); // more window axes than rank
    assert!(a.windows(&[], &[]).is_err()); // empty sizes
    assert!(a.windows(&[2], &[0]).is_err()); // zero stride
    assert!(a.windows(&[2], &[1, 1]).is_err()); // strides/sizes length mismatch
}
