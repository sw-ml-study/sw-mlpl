//! README "tour" counts vs source-of-truth checks.
//!
//! The README brags about specific counts ("17 worked
//! demos, 32 tutorial lessons, 220-entry glossary"); this
//! module is `#[cfg(test)]`-only and asserts each count is
//! still accurate. Caught the 19 -> 32 lesson drift that
//! prompted these tests after a multi-saga lesson backfill;
//! catches future drift the moment it lands.
//!
//! Source-of-truth definitions:
//! - lessons: `LESSONS.len()` (covers both `lessons.rs` and
//!   the named consts re-exported from `lessons_advanced.rs`)
//! - demos: `DEMOS.len()`
//! - glossary entries: `^## ` H2 headers in
//!   `docs/glossary.md`
//!
//! Lives in a sibling `src/readme_counts.rs` rather than
//! inline in `lessons.rs` to keep that file under the
//! file-LOC budget.

use mlpl_web_demos::DEMOS;
use mlpl_web_lessons::lessons::LESSONS;

const README: &str = include_str!("../../../../../README.md");
const GLOSSARY: &str = include_str!("../../../../../docs/glossary.md");

fn glossary_entries() -> usize {
    GLOSSARY.lines().filter(|l| l.starts_with("## ")).count()
}

#[test]
fn readme_lesson_count_matches_lessons_array() {
    let n = LESSONS.len();
    let phrase_a = format!("{n} tutorial lessons");
    let phrase_b = format!("Index of all {n} lessons");
    assert!(
        README.contains(&phrase_a),
        "README missing {phrase_a:?} (LESSONS has {n} entries)"
    );
    assert!(README.contains(&phrase_b), "README missing {phrase_b:?}");
}

#[test]
fn readme_demo_count_matches_demos_array() {
    let n = DEMOS.len();
    let phrase = format!("{n} worked");
    assert!(
        README.contains(&phrase),
        "README missing {phrase:?} (DEMOS has {n} entries)"
    );
}

#[test]
fn readme_glossary_count_matches_glossary_md() {
    let n = glossary_entries();
    let phrase = format!("{n}-entry glossary");
    assert!(
        README.contains(&phrase),
        "README missing {phrase:?} (glossary.md has {n} H2 headers)"
    );
}
