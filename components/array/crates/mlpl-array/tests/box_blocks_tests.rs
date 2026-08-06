//! Blocked box display: rank-3/4 arrays render as grids of
//! boxed inner matrices (APL2 DISPLAY of enclosed blocks,
//! from flat arrays).

use mlpl_array::{DenseArray, Shape, box_display};

fn arr(dims: Vec<usize>, n: usize) -> DenseArray {
    DenseArray::new(Shape::new(dims), (0..n).map(|i| i as f64).collect()).unwrap()
}

#[test]
fn rank4_renders_an_outer_grid_of_boxed_inner_matrices() {
    let out = box_display(&arr(vec![3, 3, 3, 3], 81));
    // Outer frame present...
    assert!(out.starts_with('+'), "outer frame:\n{out}");
    // ...three inner boxes side by side on a frame line: the
    // inner top borders appear tripled within one line.
    let inner_border_rows: Vec<&str> = out
        .lines()
        .filter(|l| l.matches("+-").count() >= 3 && !l.starts_with("+-"))
        .collect();
    assert!(
        !inner_border_rows.is_empty(),
        "three side-by-side inner frames expected:\n{out}"
    );
    // First inner matrix holds 0 1 2 / 9.. wait: consecutive
    // reshape blocks are 0..8, so its first row is 0 1 2.
    assert!(
        out.contains("| 0 1 2 |") || out.contains("0  1  2"),
        "{out}"
    );
    // Last block ends with 80.
    assert!(out.contains("80"), "{out}");
    assert!(out.contains("rank 4"), "{out}");
}

#[test]
fn rank3_renders_one_row_of_boxed_matrices() {
    let out = box_display(&arr(vec![2, 2, 3], 12));
    // Two inner boxes side by side; values 0..5 in the first,
    // 6..11 in the second.
    let first_line_with_values = out
        .lines()
        .find(|l| l.contains('0') && l.contains('6'))
        .unwrap_or("");
    assert!(
        first_line_with_values.contains("0") && first_line_with_values.contains("6"),
        "side-by-side blocks expected:\n{out}"
    );
    assert!(out.contains("rank 3"), "{out}");
}

#[test]
fn rank2_and_below_are_unchanged() {
    let out = box_display(&arr(vec![2, 3], 6));
    assert!(out.contains("| 0 1 2 |"), "{out}");
    assert!(out.contains("rank 2"), "{out}");
    let out = box_display(&arr(vec![4], 4));
    assert!(out.contains("| 0 1 2 3 |"), "{out}");
}

#[test]
fn rank5_keeps_the_labeled_stack() {
    let out = box_display(&arr(vec![2, 2, 2, 2, 2], 32));
    assert!(out.contains("[0]"), "labeled stack for rank>=5:\n{out}");
    assert!(out.contains("rank 5"), "{out}");
}
