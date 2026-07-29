//! `def u:` functions retain their raw source (incl. `#` comment
//! lines) so `:list` prints the definition AS WRITTEN -- the
//! APL2 function-listing experience (naming-and-docs saga).

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, eval_program_value, eval_source_value};
use mlpl_parser::{lex, parse};

const DEF: &str = "def u:gun_period(g) {
    # one full gun period: 30 generations
    i = 0;
    while lt(i, 30) {
        # the same one-line Life rule each step
        g = u:gen(g);
        i = i + 1
    };
    g
}";

#[test]
fn list_shows_source_with_comments() {
    let mut env = Environment::new();
    eval_source_value(DEF, &mut env).expect("def evals");
    let listed = env.list_fn("u:gun_period").expect("listed");
    assert_eq!(listed, DEF, "verbatim source, comments intact");
}

#[test]
fn reconstruction_fallback_without_source() {
    // Direct AST eval (no source attached) keeps the old behavior.
    let mut env = Environment::new();
    let prog = parse(&lex("def u:f(x) { x + 1 }").unwrap()).unwrap();
    eval_program_value(&prog, &mut env).unwrap();
    let listed = env.list_fn("u:f").expect("listed");
    assert!(listed.contains("def u:f(x)"), "{listed}");
}

#[test]
fn redefining_replaces_the_stored_source() {
    let mut env = Environment::new();
    eval_source_value("def u:f(x) { # v1\n    x }", &mut env).unwrap();
    eval_source_value("def u:f(x) { # v2\n    x + 1 }", &mut env).unwrap();
    let listed = env.list_fn("u:f").expect("listed");
    assert!(
        listed.contains("# v2") && !listed.contains("# v1"),
        "{listed}"
    );
}

#[test]
fn doc_string_still_surfaces_in_describe() {
    let mut env = Environment::new();
    eval_source_value("def u:d(x) { \"doubles x\"; x * 2 }", &mut env).unwrap();
    let desc = env.describe_fn("u:d").expect("described");
    assert!(desc.contains("doubles x"), "{desc}");
}
