//! `ask_model` unit tests, moved out of the production module
//! (spike step 013) per the move-tests-out guidance.

// `:connect set <model>` selects the model and `resolve` returns it.
// The session override short-circuits before `$OLLAMA_MODEL` and any
// network probe, so this is deterministic + offline-safe.
#[test]
fn connect_set_selects_model_for_ask() {
    let out = mlpl_repl_connect::ask_model::connect_cmd("set foo:bar", "http://127.0.0.1:1");
    assert!(out.contains("foo:bar"), "unexpected: {out}");
    assert_eq!(
        mlpl_repl_connect::ask_model::resolve("http://127.0.0.1:1"),
        "foo:bar"
    );
}
