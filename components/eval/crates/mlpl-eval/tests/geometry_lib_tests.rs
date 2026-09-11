//! Tests for examples/viz/layout.mlpl -- the storage-layout geometry
//! library (storage-layout-viz saga). Includes the real .mlpl source and
//! exercises each function against known values, so the library and these
//! expectations cannot drift.

use mlpl_eval::{Environment, EvalError, eval_program};
use mlpl_parser::{lex, parse};

const LIB: &str = include_str!("../../../../../examples/viz/layout.mlpl");

/// Evaluate the library followed by one expression; return the array value.
fn run(expr: &str) -> Result<mlpl_array::DenseArray, EvalError> {
    let src = format!("{LIB}\n{expr}");
    let stmts = parse(&lex(&src).unwrap()).unwrap();
    eval_program(&stmts, &mut Environment::new())
}

#[test]
fn block_xyz_lays_blocks_in_16x16_layers() {
    // blocks 0..4 -> a row along x; block 16 -> next z row; block 256 -> next layer y.
    let row = run("u:block_xyz(range(4))").unwrap();
    assert_eq!(
        row.data(),
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]
    );
    let wrap = run("u:block_xyz([16, 256])").unwrap();
    assert_eq!(wrap.data(), &[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn first_blocks_and_counts() {
    assert_eq!(
        run("u:first_blocks([0, 8, 128], 8)").unwrap().data(),
        &[0.0, 1.0, 16.0]
    );
    assert_eq!(
        run("u:block_counts([8, 120, 48], 8)").unwrap().data(),
        &[1.0, 15.0, 6.0]
    );
}

const SCENE: &str = "L = {region_space: [\"flash\", \"flash\", \"ebr\"], \
     region_kind: [\"header\", \"image\", \"stack\"], \
     region_start: [0, 8, 0], region_length: [8, 48, 1024]}; \
   pal = {header: [0.5, 0.5, 0.5, 1.0], image: [1.0, 0.4, 0.2, 1.0], stack: [0.9, 0.2, 0.2, 1.0]}; \
   off = {flash: [0.0, 0.0, 0.0], ebr: [3.0, 0.0, 0.0]}; \
   g = u:region_geometry(L, pal, off, 8); ";

#[test]
fn region_geometry_centers_stack_by_block_address() {
    // fb=[0,1,0], bc=[1,6,128]; y-center = fb + bc/2 = [0.5, 4, 64];
    // ebr tower base x=3. centers row i = [base_x, y_center, 0].
    let centers = run(&format!("{SCENE} record_get(g, \"centers\")?")).unwrap();
    assert_eq!(centers.shape().dims(), &[3, 3]);
    assert_eq!(
        centers.data(),
        &[0.0, 0.5, 0.0, 0.0, 4.0, 0.0, 3.0, 64.0, 0.0]
    );
}

#[test]
fn region_geometry_sizes_are_block_heights() {
    let sizes = run(&format!("{SCENE} record_get(g, \"sizes\")?")).unwrap();
    assert_eq!(
        sizes.data(),
        &[1.0, 1.0, 1.0, 1.0, 6.0, 1.0, 1.0, 128.0, 1.0]
    );
}

#[test]
fn region_geometry_colors_and_ids() {
    let colors = run(&format!("{SCENE} record_get(g, \"colors\")?")).unwrap();
    assert_eq!(colors.shape().dims(), &[3, 4]);
    assert_eq!(
        colors.data(),
        &[0.5, 0.5, 0.5, 1.0, 1.0, 0.4, 0.2, 1.0, 0.9, 0.2, 0.2, 1.0]
    );
    // No region_id column in SCENE -> positional fallback range(n).
    let ids = run(&format!("{SCENE} record_get(g, \"ids\")?")).unwrap();
    assert_eq!(ids.data(), &[0.0, 1.0, 2.0]);
}

#[test]
fn region_geometry_uses_explicit_region_id_when_present() {
    // The producer's stable region_id column is used verbatim for picking.
    let scene = "L = {region_space: [\"flash\", \"flash\", \"ebr\"], \
         region_kind: [\"header\", \"image\", \"stack\"], \
         region_start: [0, 8, 0], region_length: [8, 48, 1024], \
         region_id: [10, 20, 30]}; \
       pal = {header: [0.5, 0.5, 0.5, 1.0], image: [1.0, 0.4, 0.2, 1.0], stack: [0.9, 0.2, 0.2, 1.0]}; \
       off = {flash: [0.0, 0.0, 0.0], ebr: [3.0, 0.0, 0.0]}; \
       record_get(u:region_geometry(L, pal, off, 8), \"ids\")?";
    let ids = run(scene).unwrap();
    assert_eq!(ids.data(), &[10.0, 20.0, 30.0]);
}
