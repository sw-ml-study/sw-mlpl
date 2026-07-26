//! Game of Life engine invariance (saga acceptance test): a
//! glider on a 7x7 torus returns EXACTLY home after 28
//! generations (4-step cycle x 7 diagonal wraps) under the
//! rotate-based functional rule the demo ships.

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};

#[test]
fn glider_translates_after_28_generations() {
    // On a 7x7 torus a glider returns to its own shape translated;
    // after 28 steps (4-step cycle x 7 shifts) it is EXACTLY the
    // starting board again. Weaker but robust invariant: population
    // stays 5 at every step.
    let program = r#"
b = fill([49], 0)
b = scatter(scatter(b, 9, 1), 17, 1)
b = scatter(scatter(scatter(b, 22, 1), 23, 1), 24, 1)
G = reshape(b, [7, 7])
def u:life(g) { a = rotate(g, 1, 0); b2 = rotate(g, 0 - 1, 0); n = a + b2 + rotate(g, 1, 1) + rotate(g, 0 - 1, 1) + rotate(a, 1, 1) + rotate(a, 0 - 1, 1) + rotate(b2, 1, 1) + rotate(b2, 0 - 1, 1); gt(eq(n, 3) + g * eq(n, 2), 0) }
g = G
i = 0
while lt(i, 28) { g = u:life(g); i = i + 1 }
reduce(:and, eq(reshape(g, [49]), reshape(G, [49])))
"#;
    let toks = lex(program).expect("lex");
    let prog = parse(&toks).expect("parse");
    let mut env = Environment::new();
    let v = eval_program_value(&prog, &mut env).expect("eval");
    match v {
        mlpl_eval::Value::Array(a) => assert_eq!(a.data(), &[1.0], "glider did not return home"),
        other => panic!("expected scalar, got {other:?}"),
    }
}
