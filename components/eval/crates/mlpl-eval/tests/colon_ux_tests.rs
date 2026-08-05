//! Colon-line UX at the shared eval layer: commands typed with
//! parentheses, builtin names handed to `:describe` / `:list` /
//! `:help` in their colon-reference spelling, and unknown help
//! topics must all answer with a pointer to the right form --
//! never a bare "unknown function" or a silently evaluated line.

use mlpl_eval::{Environment, colon_fallthrough_error, inspect};

#[test]
fn command_with_parens_gets_the_no_parens_hint() {
    for line in [":history()", ":describe(x)", ":vars()"] {
        let msg =
            colon_fallthrough_error(line).unwrap_or_else(|| panic!("hint expected for {line}"));
        assert!(
            msg.contains("no parentheses"),
            "no-parens hint expected for {line}, got {msg:?}"
        );
    }
}

#[test]
fn builtin_colon_calls_still_pass_through() {
    assert_eq!(colon_fallthrough_error(":disp(x)"), None);
    assert_eq!(colon_fallthrough_error("disp(x)"), None);
    let msg = colon_fallthrough_error(":disp x").expect("hint");
    assert!(msg.contains("builtin REFERENCE"), "got {msg:?}");
}

#[test]
fn describe_accepts_the_colon_reference_spelling() {
    let mut env = Environment::new();
    let out = inspect(&mut env, ":describe :disp").expect("describe output");
    assert!(
        !out.contains("not a bound variable"),
        "':disp' must describe the builtin, got {out:?}"
    );
    assert_eq!(out, inspect(&mut env, ":describe disp").expect("plain"));
}

#[test]
fn list_of_a_builtin_points_at_describe() {
    let mut env = Environment::new();
    for line in [":list disp", ":list :disp"] {
        let out = inspect(&mut env, line).expect("list output");
        assert!(
            out.contains(":describe disp"),
            "builtin redirect expected for {line}, got {out:?}"
        );
    }
}

#[test]
fn unknown_help_topic_lists_the_topics() {
    let mut env = Environment::new();
    for line in [":help :disp", ":help disp", ":help nosuchtopic"] {
        let out = inspect(&mut env, line).expect("help output");
        assert!(
            out.contains("vars") && out.contains(":describe"),
            "topic list expected for {line}, got {out:?}"
        );
    }
}

#[test]
fn known_help_topics_still_work() {
    let mut env = Environment::new();
    let out = inspect(&mut env, ":help vars").expect("topic output");
    assert!(
        out.contains("no variables") || out.contains("bound"),
        "got {out:?}"
    );
}
