//! docs/apl2-idioms.mlpl is an EXECUTABLE document: every MLPL
//! line beside its APL2 comment must keep running end to end.

use mlpl_eval::{Environment, Value};

const DOC: &str = include_str!("../../../../../docs/apl2-idioms.mlpl");

#[test]
fn the_rosetta_document_runs_end_to_end() {
    let tokens = mlpl_parser::lex(DOC).expect("lex");
    let stmts = mlpl_parser::parse(&tokens).expect("parse");
    let mut env = Environment::new();
    let v = mlpl_eval::eval_program_value(&stmts, &mut env).expect("run");
    assert!(
        matches!(&v, Value::Result { ok: true, .. }),
        "final value must be ok(...): {v:?}"
    );
}

#[test]
fn the_document_keeps_its_two_faces() {
    // APL2 glyphs live in comments; every code line stays ASCII.
    assert!(DOC.contains('\u{2373}'), "iota glyph expected in comments");
    for (i, line) in DOC.lines().enumerate() {
        let code = line.split('#').next().unwrap_or("");
        assert!(
            code.is_ascii(),
            "line {}: code before '#' must be ASCII: {line}",
            i + 1
        );
    }
    // The not-yet-expressible section is present and plain-spoken.
    assert!(DOC.contains("Not an MLPL builtin."), "boundary markers");
}
