//! Every `Step::Demo` and `Step::Glossary` reference in a `LearningPath`
//! must resolve to a real demo / glossary entry. The path view renders a
//! silent "(not found)" fallback for a bad reference, so without this test a
//! mistyped name ships unnoticed. Walks all `PATHS` and collects every
//! dangling reference so one run reports the full set.

use mlpl_web_demos::DEMOS;
use mlpl_web_glossary::view::find_by_term;
use mlpl_web_paths_data::{PATHS, Step};

#[test]
fn paths_are_well_formed() {
    // Guards the build.rs codegen from paths.toml: a parse error or dropped
    // table would silently shrink PATHS. Every path keeps a title and at
    // least one step, and the full set survives.
    assert!(
        PATHS.len() >= 14,
        "expected >= 14 paths, got {}",
        PATHS.len()
    );
    for p in PATHS {
        assert!(!p.title.is_empty(), "a path has an empty title");
        assert!(!p.blurb.is_empty(), "path {:?} has an empty blurb", p.title);
        assert!(!p.steps.is_empty(), "path {:?} has no steps", p.title);
    }
}

#[test]
fn every_demo_and_glossary_reference_resolves() {
    let mut dangling: Vec<String> = Vec::new();
    for path in PATHS {
        for step in path.steps {
            match step {
                Step::Demo { name, .. } => {
                    if !DEMOS.iter().any(|d| d.name == *name) {
                        dangling.push(format!("{}: demo {name:?} not in DEMOS", path.title));
                    }
                }
                Step::Glossary { term, .. } if find_by_term(term).is_none() => {
                    dangling.push(format!("{}: glossary {term:?} not found", path.title));
                }
                _ => {}
            }
        }
    }
    assert!(
        dangling.is_empty(),
        "{} dangling path reference(s):\n  - {}",
        dangling.len(),
        dangling.join("\n  - ")
    );
}
