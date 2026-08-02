//! Every documented builtin must at least be MENTIONED in the
//! glossary (backticked, called, or in a heading) -- the playground
//! glossary claims to cover "every language keyword, builtin, and
//! ML concept the demos touch", and this pins that claim so new
//! builtins cannot ship glossary-invisible again (user report:
//! running_product was findable nowhere in the online docs).

#[test]
fn every_documented_builtin_is_mentioned_in_the_glossary() {
    let gloss = include_str!("../../../../../docs/glossary.md").to_lowercase();
    let missing: Vec<&str> = mlpl_eval_core::inspect_groups::documented_builtin_names()
        .filter(|name| {
            let ticked = format!("`{name}");
            let called = format!("{name}(");
            !gloss.contains(&ticked) && !gloss.contains(&called)
        })
        .collect();
    assert!(
        missing.is_empty(),
        "builtins with no glossary mention (add an entry or a \
         mention to docs/glossary.md): {missing:?}"
    );
}
