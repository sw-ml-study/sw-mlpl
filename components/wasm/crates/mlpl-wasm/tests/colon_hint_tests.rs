//! The colon-forms trichotomy at the local (browser) eval surface:
//! `:name(args)` calls the builtin, bare `:name` is a reference,
//! and `:name arg` is neither -- it must produce the same
//! trichotomy hint the server gives, never fall through to
//! evaluating the line as a program (which silently printed the
//! trailing expression's value).

use mlpl_wasm::WasmSession;

#[test]
fn colon_builtin_with_space_hints_instead_of_evaluating() {
    let s = WasmSession::new();
    s.eval("x = [1, 2, 3]");
    let out = s.eval(":disp x");
    assert!(
        out.contains("builtin REFERENCE"),
        "trichotomy hint expected, got {out:?}"
    );
    assert!(
        !out.contains("1 2 3"),
        "must not evaluate the trailing expression: {out:?}"
    );
}

#[test]
fn colon_call_still_evaluates() {
    let s = WasmSession::new();
    s.eval("x = [1, 2, 3]");
    let out = s.eval(":disp(x)");
    assert!(out.contains("rank 1"), "disp box expected, got {out:?}");
}

#[test]
fn unknown_colon_line_errors_without_evaluating() {
    let s = WasmSession::new();
    s.eval("x = 5");
    let out = s.eval(":help x");
    assert!(
        out.contains("unknown command"),
        "unknown-command error expected, got {out:?}"
    );
    assert!(!out.trim().ends_with('5'), "must not print x: {out:?}");
}
