//! Coverage for the `loss_landscape` builtin and the exact MLPL pattern the
//! "How Gradient Descent Works" demo uses: build a 2-weight MSE loss surface
//! over a grid, walk gradient descent recording the (w, b) trajectory,
//! normalize it to [0, 1] over the grid bounds, and overlay it. Kept tiny
//! (4x4 grid, 5 steps) so it runs in milliseconds on the dev profile.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run_str(src: &str) -> String {
    let toks = lex(src).expect("lex");
    let prog = parse(&toks).expect("parse");
    let mut env = Environment::default();
    match eval_program_value(&prog, &mut env).expect("eval") {
        Value::Str(s) => s,
        other => panic!("expected an SVG string, got {other:?}"),
    }
}

/// Eval every statement (one per line), returning the final value. Mirrors
/// the demo runner: a single error anywhere fails the whole program.
fn run_lines(lines: &[&str]) -> Value {
    let mut env = Environment::default();
    let mut last = Value::Array(mlpl_array::DenseArray::from_scalar(0.0));
    for line in lines {
        let toks = lex(line).expect("lex");
        let prog = parse(&toks).expect("parse");
        last = eval_program_value(&prog, &mut env).expect("eval");
    }
    last
}

#[test]
fn loss_landscape_dispatches() {
    let svg = run_str(
        "surf = [4, 3, 2, 1, 3, 2, 1, 0, 2, 1, 0, 1, 1, 0, 1, 2]\n\
         path = [[0.1, 0.9], [0.4, 0.6], [0.6, 0.3], [0.7, 0.1]]\n\
         loss_landscape(surf, [4, 4], path)",
    );
    assert!(svg.starts_with("<svg"), "svg: {svg}");
    assert_eq!(svg.matches("<polyline").count(), 1, "trajectory: {svg}");
}

#[test]
fn gradient_descent_surface_and_path_pattern_runs() {
    // The demo's exact data-flow in miniature: vectorized MSE surface over a
    // 4x4 (w, b) grid + a 5-step descent trajectory normalized to [0, 1].
    let svg = run_str(
        "x = [0, 1, 2]\n\
         y = [1, 3, 5]\n\
         G = grid([-1, 5, -3, 3], 4)\n\
         wcol = reshape(take(transpose(G), 0, 0), [16, 1])\n\
         bcol = reshape(take(transpose(G), 0, 1), [16, 1])\n\
         preds = matmul(wcol, reshape(x, [1, 3])) + matmul(bcol, ones([1, 3]))\n\
         resid = preds - matmul(ones([16, 1]), reshape(y, [1, 3]))\n\
         surf = reduce_add(resid * resid, 1) / 3\n\
         w = 4.0\n\
         b = -2.0\n\
         path = [[w, b]]\n\
         repeat 5 { pr = w * x + b; e = pr - y; dw = mean(e * x) * 2; db = mean(e) * 2; w = w - 0.05 * dw; b = b - 0.05 * db; path = concat(path, [[w, b]], 0) }\n\
         pathn = (path - matmul(ones([6, 1]), [[-1, -3]])) / matmul(ones([6, 1]), [[6, 6]])\n\
         loss_landscape(surf, [4, 4], pathn)",
    );
    assert!(svg.starts_with("<svg"), "svg: {svg}");
    assert_eq!(svg.matches("<polyline").count(), 1, "trajectory: {svg}");
    // Start (green) and end (red) trajectory markers present.
    assert!(
        svg.contains("#a6e3a1") && svg.contains("#f38ba8"),
        "markers: {svg}"
    );
}

#[test]
fn how_gradient_descent_works_demo_runs_and_converges() {
    // The full "How Gradient Descent Works" demo program (light: scalar GD,
    // no autograd/training), verified end to end. The final value is the
    // learned bias, which should land near the true 1.
    let last = run_lines(&[
        "x = linspace(-2, 2, 11)",
        "y = 2 * x + 1 + randn(5, [11]) * 0.3",
        "G = grid([-1, 5, -3, 3], 28)",
        "wcol = reshape(take(transpose(G), 0, 0), [784, 1])",
        "bcol = reshape(take(transpose(G), 0, 1), [784, 1])",
        "preds = matmul(wcol, reshape(x, [1, 11])) + matmul(bcol, ones([1, 11]))",
        "resid = preds - matmul(ones([784, 1]), reshape(y, [1, 11]))",
        "surf = reduce_add(resid * resid, 1) / 11",
        "w = 4.5",
        "b = -2.5",
        "losses = []",
        "gmags = []",
        "path = [[w, b]]",
        "repeat 40 { e = (w * x + b) - y; dw = mean(e * x) * 2; db = mean(e) * 2; losses = concat(losses, [mean(e * e)]); gmags = concat(gmags, [sqrt(dw * dw + db * db)]); w = w - 0.08 * dw; b = b - 0.08 * db; path = concat(path, [[w, b]], 0) }",
        "pathn = (path - matmul(ones([41, 1]), [[-1, -3]])) / matmul(ones([41, 1]), [[6, 6]])",
        "loss_landscape(surf, [28, 28], pathn)",
        "loss_curve(losses)",
        "svg(gmags, \"line\")",
        "bfit = b",
        "bfit",
    ]);
    match last {
        Value::Array(a) => {
            let bias = a.data()[0];
            assert!(
                (bias - 1.0).abs() < 0.6,
                "bias should converge near 1, got {bias}"
            );
        }
        other => panic!("expected scalar bias, got {other:?}"),
    }
}
