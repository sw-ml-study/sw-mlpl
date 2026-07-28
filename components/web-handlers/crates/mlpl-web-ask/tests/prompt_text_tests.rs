//! The `:ask` grounding text must pin the rules Ollama actually
//! violates in practice (observed 2026-07-28: invented keyword
//! arguments and a model built+trained inside a `def u:` body).

use mlpl_web_ask::prompt_text::mlpl_reference;

#[test]
fn reference_forbids_keyword_arguments() {
    let r = mlpl_reference();
    assert!(
        r.contains("POSITIONAL") && r.contains("NO keyword"),
        "must state args are positional-only"
    );
    assert!(
        r.contains("NOT linear(in="),
        "must show the exact hallucination to avoid"
    );
}

#[test]
fn reference_pins_top_level_only_models_and_train() {
    let r = mlpl_reference();
    assert!(
        r.contains("TOP LEVEL") && r.contains("cannot create, receive, or train a model"),
        "must state the model/train scoping rule"
    );
}

#[test]
fn reference_includes_a_canonical_training_example() {
    let r = mlpl_reference();
    assert!(
        r.contains("m = chain(linear(") && r.contains("adam(cross_entropy(apply(m, X), Y), m,"),
        "must carry one correct end-to-end training example"
    );
}

#[test]
fn reference_still_carries_the_builtin_signatures() {
    let r = mlpl_reference();
    assert!(r.contains("linear(in, out, seed)"), "signatures intact");
}
