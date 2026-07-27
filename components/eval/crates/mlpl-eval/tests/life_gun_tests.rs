//! Gosper glider gun dynamics (glider-gun saga): the canonical
//! 36-cell gun on a 40x40 torus emits one glider per 30
//! generations -- population 36 -> 41 at gen 30 -> 46 at gen 60.

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<mlpl_eval::Value, String> {
    let prog =
        parse(&lex(src).map_err(|e| format!("lex {e:?}"))?).map_err(|e| format!("parse {e:?}"))?;
    eval_program_value(&prog, env).map_err(|e| format!("eval {e:?}"))
}

fn pop(env: &mut Environment, var: &str) -> f64 {
    match run(env, &format!("reduce_add(reshape({var}, [1600]))")).expect("pop") {
        mlpl_eval::Value::Array(a) => a.data()[0],
        other => panic!("expected scalar, got {other:?}"),
    }
}

const SETUP: &[&str] = &[
    "def u:stamp(b, ids) { i = 0; while lt(i, tally(ids)) { b = scatter(b, take(ids, 0, i), 1); i = i + 1 }; b }",
    "gun = [146, 184, 186, 214, 215, 222, 223, 236, 237, 253, 257, 262, 263, 276, 277, 282, 283, 292, 298, 302, 303, 322, 323, 332, 336, 338, 339, 344, 346, 372, 378, 386, 413, 417, 454, 455]",
    "G = reshape(u:stamp(fill([1600], 0), gun), [40, 40])",
    "def u:gen(g) { a = rotate(g, 1, 0); b2 = rotate(g, 0 - 1, 0); n = a + b2 + rotate(g, 1, 1) + rotate(g, 0 - 1, 1) + rotate(a, 1, 1) + rotate(a, 0 - 1, 1) + rotate(b2, 1, 1) + rotate(b2, 0 - 1, 1); gt(eq(n, 3) + g * eq(n, 2), 0) }",
];

#[test]
fn gun_emits_one_glider_per_thirty_generations() {
    let mut env = Environment::new();
    for l in SETUP {
        run(&mut env, l).expect("setup");
    }
    assert_eq!(pop(&mut env, "G"), 36.0, "canonical gun population");
    run(
        &mut env,
        "g = G; i = 0; while lt(i, 30) { g = u:gen(g); i = i + 1 }",
    )
    .expect("30 gens");
    assert_eq!(pop(&mut env, "g"), 41.0, "gun + first glider");
    run(
        &mut env,
        "i = 0; while lt(i, 30) { g = u:gen(g); i = i + 1 }",
    )
    .expect("60 gens");
    assert_eq!(pop(&mut env, "g"), 46.0, "gun + two gliders");
}
