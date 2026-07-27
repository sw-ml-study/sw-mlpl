//! Life Pattern Zoo dynamics (life-depth saga step 002): on the
//! 20x20 torus the seeded lifeforms behave canonically -- the
//! blinker oscillates with period 2, the block is a still life,
//! and the lone pair starves in one generation.

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<mlpl_eval::Value, String> {
    let prog = parse(&lex(src).map_err(|e| format!("lex {e:?}"))?).map_err(|e| format!("parse {e:?}"))?;
    eval_program_value(&prog, env).map_err(|e| format!("eval {e:?}"))
}

// 20x20 torus; flat index = r*20 + c.
// blinker at rows 2, col 2-4 (row 2: 42,43,44)
// block (still) at (2,10): 50,51,70,71  -> r2c10=50? r*20+c: 2*20+10=50, 51; 3*20+10=70,71
// glider at (8,2): cells (8,3),(9,4),(10,2),(10,3),(10,4) -> 163,184,202,203,204
// r-pentomino at (8,13): (8,14),(8,15),(9,13),(9,14),(10,14) -> 174,175,193,194,214
// vanishing pair at (15,5),(15,6) -> 305,306 (two neighbors die next gen)
const LINES: &[&str] = &[
    "def u:stamp(b, ids) { i = 0; while lt(i, tally(ids)) { b = scatter(b, take(ids, 0, i), 1); i = i + 1 }; b }",
    "b = u:stamp(fill([400], 0), [42, 43, 44])",
    "b = u:stamp(b, [50, 51, 70, 71])",
    "b = u:stamp(b, [163, 184, 202, 203, 204])",
    "b = u:stamp(b, [174, 175, 193, 194, 214])",
    "b = u:stamp(b, [305, 306])",
    "Z = reshape(b, [20, 20])",
    "disp(Z)",
    "def u:life20(g) { a = rotate(g, 1, 0); b2 = rotate(g, 0 - 1, 0); n = a + b2 + rotate(g, 1, 1) + rotate(g, 0 - 1, 1) + rotate(a, 1, 1) + rotate(a, 0 - 1, 1) + rotate(b2, 1, 1) + rotate(b2, 0 - 1, 1); gt(eq(n, 3) + g * eq(n, 2), 0) }",
    "g1 = u:life20(Z)",
    "reduce_add(reshape(g1, [400]))",
    "disp(reshape(concat(reshape(Z, [1, 20, 20]), reshape(g1, [1, 20, 20]), 0), [2, 20, 20]))",
    "F = reshape(Z, [1, 20, 20])",
    "g = Z",
    "i = 0",
    "while lt(i, 31) { g = u:life20(g); emit_frame(\"life\", i, g); F = concat(F, reshape(g, [1, 20, 20]), 0); i = i + 1 }",
    "shape(F)",
    "svg(F, \"life\")",
];

#[test]
fn probe() {
    let mut env = Environment::new();
    for (i, line) in LINES.iter().enumerate() {
        match run(&mut env, line) {
            Ok(v) => {
                let s = format!("{v}");
                eprintln!("OK [{i}] {line}\n{}", &s[..s.len().min(300)]);
            }
            Err(e) => panic!("[{i}] {line}\n  {e}"),
        }
    }
}

#[test]
fn zoo_dynamics_hold() {
    // After 2 generations: blinker oscillates (period 2, back to start),
    // block unchanged, vanishing pair gone.
    let mut env = Environment::new();
    for l in &LINES[..9] {
        run(&mut env, l).expect("setup");
    }
    run(&mut env, "g2 = u:life20(u:life20(Z))").expect("two gens");
    // block cells still alive
    let v = run(&mut env, "take(reshape(g2, [400]), 0, 50) + take(reshape(g2, [400]), 0, 71)").expect("block");
    assert_eq!(format!("{v}"), "2", "block is a still life");
    // vanishing pair dead by gen 1
    run(&mut env, "g1b = u:life20(Z)").expect("one gen");
    let v = run(&mut env, "take(reshape(g1b, [400]), 0, 305) + take(reshape(g1b, [400]), 0, 306)").expect("pair");
    assert_eq!(format!("{v}"), "0", "lone pair starves");
    // blinker back to horizontal after 2
    let v = run(&mut env, "take(reshape(g2, [400]), 0, 42) + take(reshape(g2, [400]), 0, 43) + take(reshape(g2, [400]), 0, 44)").expect("blinker");
    assert_eq!(format!("{v}"), "3", "blinker period 2");
}
