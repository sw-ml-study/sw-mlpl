//! Diagnostic analysis for the Moons MLP demo decision boundary.
//! Saga 33 step 025 investigation. Run with:
//!   cargo test -p mlpl-eval --release --test \
//!     moons_mlp_boundary_analysis -- --ignored --nocapture

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn run_and_get_p1(src: &str) -> Vec<f64> {
    let tokens = lex(src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    let mut env = Environment::default();
    let val = eval_program_value(&stmts, &mut env).expect("eval");
    match val {
        Value::Array(a) => a.data().to_vec(),
        other => panic!("demo must return Value::Array, got {other:?}"),
    }
}

const DEMO_CE: &str = r#"
M = moons(7, 120, 0.08)
X = matmul(M, [[1,0],[0,1],[0,0]])
y = reshape(matmul(M, [[0],[0],[1]]), [120])
O120 = ones([120, 1])
W1 = param[2, 8]
b1 = param[1, 8]
W2 = param[8, 2]
b2 = param[1, 2]
W1 = randn(11, [2, 8]) * 0.5
W2 = randn(12, [8, 2]) * 0.5
train 200 { adam(cross_entropy(matmul(tanh_fn(matmul(X, W1) + matmul(O120, b1)), W2) + matmul(O120, b2), y), [W1, b1, W2, b2], 0.05, 0.9, 0.999, 0.00000001); cross_entropy(matmul(tanh_fn(matmul(X, W1) + matmul(O120, b1)), W2) + matmul(O120, b2), y) }
G = grid([-1.5, 2.5, -1, 1.5], 30)
O900 = ones([900, 1])
GZ1 = matmul(G, W1) + matmul(O900, b1)
GH = tanh_fn(GZ1)
GZ2 = matmul(GH, W2) + matmul(O900, b2)
GP = softmax(GZ2, 1)
p1 = reshape(matmul(GP, [[0],[1]]), [900])
p1
"#;

const DEMO_MSE: &str = r#"
M = moons(7, 120, 0.08)
X = matmul(M, [[1,0],[0,1],[0,0]])
y = reshape(matmul(M, [[0],[0],[1]]), [120])
Y = one_hot(y, 2)
O120 = ones([120, 1])
W1 = param[2, 8]
b1 = param[1, 8]
W2 = param[8, 2]
b2 = param[1, 2]
W1 = randn(11, [2, 8]) * 0.5
W2 = randn(12, [8, 2]) * 0.5
train 200 { adam(mean((matmul(tanh_fn(matmul(X, W1) + matmul(O120, b1)), W2) + matmul(O120, b2) - Y) * (matmul(tanh_fn(matmul(X, W1) + matmul(O120, b1)), W2) + matmul(O120, b2) - Y)), [W1, b1, W2, b2], 0.05, 0.9, 0.999, 0.00000001); mean((matmul(tanh_fn(matmul(X, W1) + matmul(O120, b1)), W2) + matmul(O120, b2) - Y) * (matmul(tanh_fn(matmul(X, W1) + matmul(O120, b1)), W2) + matmul(O120, b2) - Y)) }
G = grid([-1.5, 2.5, -1, 1.5], 30)
O900 = ones([900, 1])
GZ1 = matmul(G, W1) + matmul(O900, b1)
GH = tanh_fn(GZ1)
GZ2 = matmul(GH, W2) + matmul(O900, b2)
GP = softmax(GZ2, 1)
p1 = reshape(matmul(GP, [[0],[1]]), [900])
p1
"#;

const DEMO_DSL: &str = r#"
M = moons(7, 120, 0.08)
X = matmul(M, [[1,0],[0,1],[0,0]])
y = reshape(matmul(M, [[0],[0],[1]]), [120])
Y = one_hot(y, 2)
mdl = chain(linear(2, 8, 11), tanh_layer(), linear(8, 2, 12))
train 200 { adam(mean((apply(mdl, X) - Y) * (apply(mdl, X) - Y)), mdl, 0.05, 0.9, 0.999, 0.00000001); mean((apply(mdl, X) - Y) * (apply(mdl, X) - Y)) }
G = grid([-1.5, 2.5, -1, 1.5], 30)
GP = softmax(apply(mdl, G), 1)
p1 = reshape(matmul(GP, [[0],[1]]), [900])
p1
"#;

fn print_grid_ascii(label: &str, p1: &[f64]) {
    println!("\n{label} -- p1 grid (30x30, # = class 1 i.e. high p1, . = class 0):");
    println!(
        "    {}",
        (0..30).map(|c| (c % 10).to_string()).collect::<String>()
    );
    for r in 0..30 {
        let row: String = (0..30)
            .map(|c| if p1[r * 30 + c] > 0.5 { '#' } else { '.' })
            .collect();
        println!("{r:2}: {row}");
    }
}

fn boundary_stats(label: &str, p1: &[f64]) {
    let n = 30;
    let mean = p1.iter().sum::<f64>() / 900.0;
    let total_var = p1.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 900.0;
    let mut row_means = vec![0.0; n];
    let mut col_means = vec![0.0; n];
    for r in 0..n {
        for c in 0..n {
            row_means[r] += p1[r * n + c];
            col_means[c] += p1[r * n + c];
        }
    }
    for v in row_means.iter_mut() {
        *v /= n as f64;
    }
    for v in col_means.iter_mut() {
        *v /= n as f64;
    }
    let row_mean_var = row_means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / n as f64;
    let col_mean_var = col_means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / n as f64;
    println!("\n{label} stats:");
    println!("  mean(p1)      = {mean:.4}");
    println!("  total_var     = {total_var:.4}");
    println!(
        "  row_mean_var  = {:.4}  ({:>2}% of total) -- variance ACROSS rows (y-sensitivity)",
        row_mean_var,
        (row_mean_var / total_var * 100.0) as i32
    );
    println!(
        "  col_mean_var  = {:.4}  ({:>2}% of total) -- variance ACROSS cols (x-sensitivity)",
        col_mean_var,
        (col_mean_var / total_var * 100.0) as i32
    );
    println!(
        "  ratio col/row = {:.2}  (>5: vertical band; <0.2: horizontal band; ~1: curved)",
        col_mean_var / row_mean_var.max(1e-9)
    );
    println!("  corner samples (grid (r, c) -> p1):");
    for &(r, c, name) in &[
        (0u32, 0u32, "image TL  math(xmin, ymin)"),
        (0, 29, "image TR  math(xmax, ymin)"),
        (29, 0, "image BL  math(xmin, ymax)"),
        (29, 29, "image BR  math(xmax, ymax)"),
    ] {
        println!(
            "    ({:2},{:2})  p1={:.4}  {}",
            r,
            c,
            p1[(r * 30 + c) as usize],
            name
        );
    }
}

#[test]
#[ignore = "diagnostic only; run with --ignored"]
fn analyze_moons_boundary_all_three_variants() {
    let p1_ce = run_and_get_p1(DEMO_CE);
    let p1_mse = run_and_get_p1(DEMO_MSE);
    let p1_dsl = run_and_get_p1(DEMO_DSL);

    print_grid_ascii("CROSS_ENTROPY (current web demo)", &p1_ce);
    boundary_stats("CROSS_ENTROPY", &p1_ce);

    print_grid_ascii("MSE on logits (raw-matmul, MSE loss)", &p1_mse);
    boundary_stats("MSE", &p1_mse);

    print_grid_ascii("MODEL DSL (chain+apply, MSE)", &p1_dsl);
    boundary_stats("DSL", &p1_dsl);
}
