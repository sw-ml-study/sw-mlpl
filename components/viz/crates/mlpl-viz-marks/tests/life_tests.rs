//! `svg(frames, "life")` SMIL grid animation (Game of Life saga
//! step 2). One self-contained SVG that steps through the [T,H,W]
//! frames with discrete-mode opacity animations; T=1 (or rank-2
//! input) degrades to a static grid with no animation.

use mlpl_array::{DenseArray, Shape};
use mlpl_viz_marks::render_life;

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

/// Vertical blinker then horizontal blinker on a 5x5 board.
fn blinker_frames() -> DenseArray {
    let mut f0 = vec![0.0; 25];
    for r in 1..4 {
        f0[r * 5 + 2] = 1.0;
    }
    let mut f1 = vec![0.0; 25];
    for c in 1..4 {
        f1[2 * 5 + c] = 1.0;
    }
    let data: Vec<f64> = f0.into_iter().chain(f1).collect();
    arr(vec![2, 5, 5], data)
}

#[test]
fn blinker_two_frames_animate() {
    let svg = render_life(&blinker_frames()).unwrap();
    assert_eq!(svg.matches("<g class=\"life-frame\"").count(), 2, "{svg}");
    assert_eq!(svg.matches("<animate ").count(), 2, "{svg}");
    assert!(svg.contains("repeatCount=\"indefinite\""), "{svg}");
    assert!(svg.contains("calcMode=\"discrete\""), "{svg}");
    // 3 alive cells per frame + 1 background rect
    assert_eq!(svg.matches("<rect").count(), 1 + 3 + 3, "{svg}");
    assert!(svg.trim_start().starts_with("<svg"), "{svg}");
    assert!(svg.trim_end().ends_with("</svg>"), "{svg}");
}

#[test]
fn single_frame_is_static() {
    let one = arr(vec![1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let svg = render_life(&one).unwrap();
    assert_eq!(svg.matches("<g class=\"life-frame\"").count(), 1);
    assert!(!svg.contains("<animate"), "{svg}");
}

#[test]
fn rank2_board_treated_as_one_frame() {
    let board = arr(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]);
    let svg = render_life(&board).unwrap();
    assert_eq!(svg.matches("<g class=\"life-frame\"").count(), 1);
    assert!(!svg.contains("<animate"), "{svg}");
}

#[test]
fn wrong_rank_is_an_error() {
    assert!(render_life(&arr(vec![4], vec![0.0; 4])).is_err());
    assert!(render_life(&arr(vec![1, 1, 2, 2], vec![0.0; 4])).is_err());
}
