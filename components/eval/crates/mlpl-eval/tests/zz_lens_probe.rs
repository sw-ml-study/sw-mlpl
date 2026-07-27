use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};

fn run(env: &mut Environment, src: &str) -> Result<mlpl_eval::Value, String> {
    let toks = lex(src).map_err(|e| format!("lex {e:?}"))?;
    let prog = parse(&toks).map_err(|e| format!("parse {e:?}"))?;
    eval_program_value(&prog, env).map_err(|e| format!("eval {e:?}"))
}

#[test]
fn probe() {
    let mut env = Environment::new();
    for src in [
        "T = reshape(iota(12), [2, 2, 3])",
        "disp(T)",
        "plane = take(T, 0, 1)",
        "disp(plane)",
        "row = take(plane, 0, 0)",
        "cell = take(row, 0, 2)",
        "disp(cell)",
        "T2 = reshape(scatter(reshape(T, [12]), 11, 99), [2, 2, 3])",
        "disp(take(T2, 0, 1))",
        "disp(take(T, 0, 1))",
        "def u:get_safe(x, i) { if lt(i, tally(x)) { ok(take(x, 0, i)) } else { err({kind: \"index\", at: i, size: tally(x)}) } }",
        "r1 = u:get_safe(T, 1)",
        "is_ok(r1)",
        "bad = u:get_safe(T, 5)",
        "err_message(bad)",
        "unwrap_or(u:get_safe(T, 5), fill([2, 3], 0))",
    ] {
        match run(&mut env, src) {
            Ok(v) => eprintln!("OK  {src}\n    => {v}"),
            Err(e) => panic!("FAILED {src}\n    {e}"),
        }
    }
    // the intentional hard-error line
    let err = run(&mut env, "take(T, 0, 5)").expect_err("OOB take must hard-error");
    eprintln!("HARD ERROR (as designed): {err}");
}
