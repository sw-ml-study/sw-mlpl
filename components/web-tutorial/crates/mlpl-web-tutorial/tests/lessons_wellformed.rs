//! Guards the build.rs codegen of LESSONS from lessons.toml: a parse error or
//! a dropped table would silently shrink the tutorial. Every lesson keeps a
//! title, intro, and at least one example, and the full set survives.

use mlpl_web_tutorial::LESSONS;

#[test]
fn lessons_are_well_formed() {
    assert!(
        LESSONS.len() >= 57,
        "expected >= 57 lessons, got {}",
        LESSONS.len()
    );
    for l in LESSONS {
        assert!(!l.title.is_empty(), "a lesson has an empty title");
        assert!(!l.intro.is_empty(), "lesson {:?} has an empty intro", l.title);
        assert!(
            !l.examples.is_empty(),
            "lesson {:?} has no examples",
            l.title
        );
        assert!(!l.try_it.is_empty(), "lesson {:?} has no try-it", l.title);
    }
}
