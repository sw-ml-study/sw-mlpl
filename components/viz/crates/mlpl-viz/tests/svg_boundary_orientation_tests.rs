//! Regression tests for saga 33 step 025: the boundary renderers
//! must use the same y convention as their overlaid scatter points
//! (math y_min at image bottom). The pre-fix renderers iterated
//! `raw[r*cols+c]` with r=0 at image top, which silently flipped
//! the surface upside-down relative to its own training points --
//! the "Moons MLP out-of-phase" bug.
//!
//! These tests use a SYNTHETIC grid where the value depends only
//! on the row index (so the test can read the SVG colors top-to-bottom
//! and assert the orientation), independent of the live demo path.

use mlpl_array::{DenseArray, Shape};
use mlpl_viz::{analysis_boundary_2d, render_decision_boundary};

fn matrix(rows: usize, cols: usize, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(vec![rows, cols]), data).unwrap()
}

/// Build a `rows`x`cols` grid whose value is the row index
/// normalized to `[0, 1]` (`v = r / (rows-1)`). Same value across
/// all columns within a row -- so the rendered surface has
/// constant color per row, varying with row.
fn row_indexed_grid(rows: usize, cols: usize) -> DenseArray {
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let t = r as f64 / (rows as f64 - 1.0);
        for _ in 0..cols {
            data.push(t);
        }
    }
    matrix(rows, cols, data)
}

/// Extract every `<rect ... fill="..."/>` element in the SVG that
/// sits inside the plot area (skipping background + colorbar
/// rects). Returns a flat list of `(y, fill)` pairs sorted by
/// `y`. With a row-indexed grid the values should monotonically
/// vary in fill across these y positions.
fn plot_rect_fills_by_y(svg: &str) -> Vec<(f64, String)> {
    // Cheap & robust SVG scan: every rect with a plot-cell width
    // (8.0 < width < 200.0) gets considered. Background rect uses
    // width=100% so it's skipped automatically. Legend colorbar
    // rects are width=10.0 (boundary_2d uses no legend, but
    // decision_boundary has 32 of them) -- exclude width <= 10.5.
    let mut out = Vec::new();
    for cap in svg.split("<rect ").skip(1) {
        let line = cap.split("/>").next().unwrap_or("");
        let get = |k: &str| -> Option<f64> {
            line.split(&format!("{k}=\""))
                .nth(1)?
                .split('"')
                .next()?
                .parse::<f64>()
                .ok()
        };
        let (Some(y), Some(w), Some(fill_start)) =
            (get("y"), get("width"), line.find("fill=\"").map(|i| i + 6))
        else {
            continue;
        };
        if w <= 10.5 || w >= 200.0 {
            continue;
        }
        let fill = line[fill_start..]
            .split('"')
            .next()
            .unwrap_or("")
            .to_string();
        out.push((y, fill));
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out
}

/// `t = raw[r=0..rows] / (rows-1)` is the color-ramp `t`. For
/// `boundary_2d`'s blue->pink ramp:
///   t=0   -> rgb(137,180,250) (blue)
///   t=1   -> rgb(243,139,168) (pink-ish)
///
/// After the saga 33 step 025 fix, t=0 (grid row 0, math ymin) lives
/// at the BOTTOM of the image (large y), and t=1 (grid row N-1, math
/// ymax) lives at the TOP (small y). So the first rect in plot-y
/// order (smallest y, top of image) should be NEAR PINK and the last
/// (bottom of image) NEAR BLUE.
#[test]
fn boundary_2d_renders_grid_row_0_at_image_bottom() {
    let grid = row_indexed_grid(5, 5);
    let pts = matrix(1, 2, vec![0.5, 0.5]);
    let labels = DenseArray::new(Shape::vector(1), vec![0.0]).unwrap();
    let svg = analysis_boundary_2d(&grid, &shape_dims_arr(5, 5), &pts, &labels).unwrap();
    let rects = plot_rect_fills_by_y(&svg);
    assert!(
        rects.len() >= 25,
        "expected >=25 plot rects, got {}",
        rects.len()
    );
    // Top-of-image rect (smallest y) -- corresponds to grid row 4
    // (math ymax, t=1.0) -- should be pink-ish (high red).
    let (top_y, top_fill) = &rects[0];
    let (bot_y, bot_fill) = rects.last().unwrap();
    let top_red = parse_red(top_fill);
    let bot_red = parse_red(bot_fill);
    assert!(
        top_red > bot_red + 50,
        "expected top (y={top_y}) to be redder than bottom (y={bot_y}); \
         top_fill={top_fill}, bot_fill={bot_fill}, top_red={top_red}, \
         bot_red={bot_red}. Before saga 33 step 025 fix, the surface was \
         vertically flipped relative to its overlaid points."
    );
}

#[test]
fn decision_boundary_renders_grid_row_0_at_image_bottom() {
    let grid = row_indexed_grid(5, 5);
    let train = matrix(1, 3, vec![0.5, 0.5, 0.0]);
    let svg = render_decision_boundary(&grid, &train).unwrap();
    let rects = plot_rect_fills_by_y(&svg);
    assert!(
        rects.len() >= 25,
        "expected >=25 plot rects, got {}",
        rects.len()
    );
    let (top_y, top_fill) = &rects[0];
    let (bot_y, bot_fill) = rects.last().unwrap();
    let top_red = parse_red(top_fill);
    let bot_red = parse_red(bot_fill);
    assert!(
        top_red > bot_red + 50,
        "expected top (y={top_y}) redder than bottom (y={bot_y}); \
         top_fill={top_fill}, bot_fill={bot_fill}, top_red={top_red}, \
         bot_red={bot_red}. Before saga 33 step 025 fix, the surface \
         was vertically flipped relative to its overlaid points."
    );
}

fn parse_red(fill: &str) -> i32 {
    // fill format: "rgb(R,G,B)"
    let s = fill.trim_start_matches("rgb(").trim_end_matches(')');
    s.split(',')
        .next()
        .and_then(|x| x.parse().ok())
        .unwrap_or(-1)
}

fn shape_dims_arr(rows: usize, cols: usize) -> DenseArray {
    DenseArray::new(Shape::vector(2), vec![rows as f64, cols as f64]).unwrap()
}
