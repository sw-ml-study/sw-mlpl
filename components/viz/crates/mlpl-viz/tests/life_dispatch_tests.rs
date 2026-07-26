//! `svg(x, "life")` reaches the marks renderer through the
//! mlpl-viz dispatch table (Game of Life saga step 2).

use mlpl_array::{DenseArray, Shape};

#[test]
fn life_type_name_dispatches() {
    let data = DenseArray::new(
        Shape::new(vec![2, 2, 2]),
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    )
    .unwrap();
    let svg = mlpl_viz::render(&data, "life").unwrap();
    assert!(svg.contains("life-frame"), "{svg}");
    assert!(svg.contains("repeatCount=\"indefinite\""), "{svg}");
}
