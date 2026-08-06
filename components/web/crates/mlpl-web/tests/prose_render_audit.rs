//! Registry-wide guard for the prose-vs-markup class bug: every
//! text that flows through the markdown-ish inline renderer
//! (demo intros/takeaways and line comments, lesson bodies,
//! glossary entries) must produce ZERO `_emph_` spans -- prose
//! here never uses underscore emphasis on purpose, so any span
//! is a snake_case identifier being eaten by markup.

use mlpl_web_demos::DEMOS;
use mlpl_web_lessons::lessons::LESSONS;
use mlpl_web_path_body::emphasis_spans;

const GLOSSARY: &str = include_str!("../../../../../docs/glossary.md");

fn assert_clean(what: &str, text: &str) {
    let hits = emphasis_spans(text);
    assert!(
        hits.is_empty(),
        "{what}: underscore emphasis would eat identifiers: {hits:?}\nin: {text}"
    );
}

#[test]
fn demo_prose_renders_without_eaten_underscores() {
    for d in DEMOS {
        assert_clean(&format!("demo {:?} intro", d.name), d.intro);
        assert_clean(&format!("demo {:?} takeaway", d.name), d.takeaway);
        for line in d.lines {
            assert_clean(&format!("demo {:?} line", d.name), line);
        }
    }
}

#[test]
fn lesson_prose_renders_without_eaten_underscores() {
    for l in LESSONS {
        assert_clean(&format!("lesson {:?} intro", l.title), l.intro);
        for ex in l.examples {
            assert_clean(&format!("lesson {:?} example", l.title), ex);
        }
        assert_clean(&format!("lesson {:?} try_it", l.title), l.try_it);
    }
}

#[test]
fn glossary_prose_renders_without_eaten_underscores() {
    for (i, line) in GLOSSARY.lines().enumerate() {
        assert_clean(&format!("glossary line {}", i + 1), line);
    }
}
