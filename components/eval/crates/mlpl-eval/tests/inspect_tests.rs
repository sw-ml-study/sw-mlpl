//! Tests for the REPL introspection commands (`:vars`, `:models`,
//! `:fns`, `:wsid`, `:describe <name>`). Lives as an integration
//! test so the `inspect` module itself stays within the project's
//! per-module function-count budget.

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, eval_program, inspect};
use mlpl_parser::{lex, parse};

fn eval(src: &str, env: &mut Environment) {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap();
}

#[test]
fn inspect_returns_none_for_non_colon() {
    let mut env = Environment::new();
    assert!(inspect(&mut env, "1 + 2").is_none());
}

#[test]
fn inspect_returns_none_for_unknown_colon() {
    let mut env = Environment::new();
    assert!(inspect(&mut env, ":unknown").is_none());
}

#[test]
fn vars_lists_arrays_with_shape_and_param_tag() {
    let mut env = Environment::new();
    env.set("x".into(), DenseArray::from_vec(vec![1.0, 2.0, 3.0]));
    env.set_param(
        "W".into(),
        DenseArray::new(Shape::new(vec![2, 3]), vec![0.0; 6]).unwrap(),
    );
    let out = inspect(&mut env, ":vars").unwrap();
    assert!(out.contains("W: [2, 3] [param]"), "out was: {out}");
    assert!(out.contains("x: [3]"), "out was: {out}");
}

#[test]
fn vars_shows_labeled_shape_for_labeled_arrays() {
    let mut env = Environment::new();
    env.set(
        "x".into(),
        DenseArray::new(Shape::new(vec![6, 4]), vec![0.0; 24])
            .unwrap()
            .with_labels(vec![Some("seq".into()), Some("d_model".into())])
            .unwrap(),
    );
    let out = inspect(&mut env, ":vars").unwrap();
    assert!(
        out.contains("x: [seq=6, d_model=4]"),
        "expected labeled shape, out was: {out}"
    );
}

#[test]
fn describe_array_shows_labeled_shape() {
    let mut env = Environment::new();
    env.set(
        "x".into(),
        DenseArray::from_vec(vec![1.0, 2.0, 3.0])
            .with_labels(vec![Some("seq".into())])
            .unwrap(),
    );
    let out = inspect(&mut env, ":describe x").unwrap();
    assert!(
        out.contains("shape: [seq=3]"),
        "expected labeled shape, out was: {out}"
    );
}

#[test]
fn describe_array_unlabeled_unchanged() {
    let mut env = Environment::new();
    env.set("x".into(), DenseArray::from_vec(vec![1.0, 2.0, 3.0]));
    let out = inspect(&mut env, ":describe x").unwrap();
    // Positional shape preserved -- no regression.
    assert!(out.contains("shape: [3]"), "out was: {out}");
}

#[test]
fn vars_partial_labels_render_mixed() {
    let mut env = Environment::new();
    env.set(
        "X".into(),
        DenseArray::new(Shape::new(vec![6, 4]), vec![0.0; 24])
            .unwrap()
            .with_labels(vec![None, Some("d_model".into())])
            .unwrap(),
    );
    let out = inspect(&mut env, ":vars").unwrap();
    assert!(
        out.contains("X: [6, d_model=4]"),
        "expected partial labeling, out was: {out}"
    );
}

#[test]
fn wsid_counts_match_env_state() {
    let mut env = Environment::new();
    env.set("a".into(), DenseArray::from_scalar(1.0));
    env.set_param("b".into(), DenseArray::from_scalar(2.0));
    let out = inspect(&mut env, ":wsid").unwrap();
    assert!(out.contains("variables:       2"), "out was: {out}");
    assert!(out.contains("parameters:      1"), "out was: {out}");
}

#[test]
fn describe_unknown_name_reports_clearly() {
    let mut env = Environment::new();
    let out = inspect(&mut env, ":describe nope").unwrap();
    assert!(out.contains("not a bound variable"), "out was: {out}");
}

#[test]
fn describe_builtin_prints_signature() {
    let mut env = Environment::new();
    let out = inspect(&mut env, ":describe softmax").unwrap();
    assert!(out.contains("softmax(a, axis)"), "out was: {out}");
}

#[test]
fn builtins_lists_model_dsl_entries() {
    let mut env = Environment::new();
    let out = inspect(&mut env, ":builtins").unwrap();
    assert!(out.contains("Model DSL"), "out was: {out}");
    assert!(out.contains("chain(a, b, ...)"), "out was: {out}");
}

#[test]
fn fns_reports_no_user_functions_yet() {
    let mut env = Environment::new();
    let out = inspect(&mut env, ":fns").unwrap();
    assert!(out.contains("no user-defined functions"), "out was: {out}");
    // The empty-state hint now points at how to DEFINE a user function
    // (more useful here than the old `:builtins` cross-reference).
    assert!(out.contains("def u:name"), "out was: {out}");
}

#[test]
fn help_topic_aliases_dispatch_to_inspectors() {
    let mut env = Environment::new();
    eval("x = [1, 2, 3]", &mut env);
    let vars = inspect(&mut env, ":help vars").unwrap();
    assert!(vars.contains("x: [3]"), "out was: {vars}");
    let builtins = inspect(&mut env, ":help builtins").unwrap();
    assert!(builtins.contains("Model DSL"), "out was: {builtins}");
    let fns = inspect(&mut env, ":help fns").unwrap();
    assert!(fns.contains("no user-defined functions"), "out was: {fns}");
}

#[test]
fn describe_model_prints_layer_tree_and_param_shapes() {
    let mut env = Environment::new();
    eval(
        "mdl = chain(linear(2, 3, 7), tanh_layer(), linear(3, 2, 8))",
        &mut env,
    );
    let out = inspect(&mut env, ":describe mdl").unwrap();
    assert!(
        out.contains("chain(linear -> tanh -> linear)"),
        "out was: {out}"
    );
    assert!(out.contains("[2, 3]"), "out was: {out}");
    assert!(out.contains("[3, 2]"), "out was: {out}");
}

#[test]
fn models_lists_bound_models() {
    let mut env = Environment::new();
    eval("lin = linear(4, 4, 9)", &mut env);
    let out = inspect(&mut env, ":models").unwrap();
    assert!(out.contains("lin: linear"), "out was: {out}");
    assert!(out.contains("(2 params)"), "out was: {out}");
}

#[test]
fn introspect_bundles_every_no_arg_section() {
    // Saga 33 step 037d: :introspect concatenates every
    // no-arg inspector under a `## :<topic>` header. Should
    // contain at minimum the seven section headers in
    // fixed order: version, wsid, builtins, vars, models,
    // experiments, tags.
    let mut env = Environment::new();
    eval("x = [1, 2, 3]", &mut env);
    eval("lin = linear(4, 4, 9)", &mut env);
    let out = inspect(&mut env, ":introspect").unwrap();
    for header in [
        "## :version",
        "## :wsid",
        "## :builtins",
        "## :vars",
        "## :models",
        "## :experiments",
        "## :tags",
    ] {
        assert!(out.contains(header), "missing header {header}");
    }
    // The bundle should embed the same content the individual
    // commands return.
    assert!(out.contains("x: [3]"), "missing :vars content");
    assert!(out.contains("lin: linear"), "missing :models content");
    // Header order is fixed -- ":version" comes before ":vars".
    let v_pos = out.find("## :version").unwrap();
    let vars_pos = out.find("## :vars").unwrap();
    assert!(v_pos < vars_pos, "section order broken");
}
