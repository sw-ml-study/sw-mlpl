//! Tests for examples/viz/memory_map_2d.mlpl -- the self-contained 2D
//! orthographic memory-map (storage-layout-viz saga). Includes the real
//! .mlpl source (its trailing svg(...) demo returns a string, harmless as a
//! statement) and checks the bucketize + grid assembly against known values.

use mlpl_array::Shape;
use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

const LIB: &str = include_str!("../../../../../examples/viz/memory_map_2d.mlpl");

fn run(expr: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let src = format!("{LIB}\n{expr}");
    let stmts = parse(&lex(&src).unwrap()).unwrap();
    eval_program(&stmts, &mut Environment::new())
}

#[test]
fn block_region_buckets_blocks_by_start() {
    // start_blocks [0,8,20]: blocks 0..7 -> region 0, 8..19 -> 1, 20..23 -> 2.
    let r = run("u:block_region([0, 8, 20], 24)").unwrap();
    let mut expected = vec![0.0; 8];
    expected.extend(vec![1.0; 12]);
    expected.extend(vec![2.0; 4]);
    assert_eq!(r.data(), &expected[..]);
}

#[test]
fn memory_heatmap_grid_shape_and_padding() {
    // starts [0,1,16,21], 26 blocks, 16 wide -> pad to 32 -> [2,16].
    let g = run("u:memory_heatmap([0, 1, 16, 21], 26, 16)").unwrap();
    assert_eq!(g.shape(), &Shape::new(vec![2, 16]));
    let row0: Vec<f64> = std::iter::once(0.0).chain(vec![1.0; 15]).collect();
    let mut row1 = vec![2.0; 5];
    row1.extend(vec![3.0; 5]);
    row1.extend(vec![-1.0; 6]);
    let mut expected = row0;
    expected.extend(row1);
    assert_eq!(g.data(), &expected[..]);
}
