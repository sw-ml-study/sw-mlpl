use mlpl_array::{DenseArray, Shape};
use mlpl_viz::render;

/// One unit block (`<rect rx="2"`) per game, colored by category.
fn block_count(svg: &str) -> usize {
    svg.matches("rx=\"2\"").count()
}

#[test]
fn waffle_vector_draws_one_block_per_game() {
    // 3 losses + 2 ties + 5 wins = 10 blocks.
    let counts = DenseArray::from_vec(vec![3.0, 2.0, 5.0]);
    let svg = render(&counts, "waffle").unwrap();
    assert!(svg.starts_with("<svg"));
    assert_eq!(block_count(&svg), 10);
    // loss=red, tie=gray, win=green all present.
    assert!(svg.contains("#f38ba8"));
    assert!(svg.contains("#6c7086"));
    assert!(svg.contains("#a6e3a1"));
}

#[test]
fn waffle_matrix_stacks_before_after_bands() {
    // Row 0 = before (6 losses, 1 tie, 3 wins), row 1 = after.
    let m = DenseArray::new(Shape::new(vec![2, 3]), vec![6.0, 1.0, 3.0, 3.0, 0.0, 7.0]).unwrap();
    let svg = render(&m, "waffle").unwrap();
    assert_eq!(block_count(&svg), 20, "10 blocks per band, two bands");
}

#[test]
fn waffle_all_wins_is_all_green() {
    let counts = DenseArray::from_vec(vec![0.0, 0.0, 8.0]);
    let svg = render(&counts, "waffle").unwrap();
    assert_eq!(block_count(&svg), 8);
    assert!(!svg.contains("#f38ba8"), "no losses -> no red");
}
