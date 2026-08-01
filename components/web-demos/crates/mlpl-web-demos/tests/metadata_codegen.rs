//! Guards the build.rs codegen of the per-demo metadata consts from
//! `demos.toml`: a parse error or a dropped `[[capabilities]]` /
//! `[[literate]]` / `[[progress_notes]]` table would silently un-gate a GPU
//! demo, drop a walkthrough link, or lose a long-run heads-up. Pin the counts
//! and spot-check that name+line lookups still resolve.

use mlpl_web_demos::{DEMO_CAPABILITIES, LITERATE_DOCS, PROGRESS_NOTES, progress_notes_for};

#[test]
fn metadata_tables_survive_codegen() {
    assert_eq!(
        DEMO_CAPABILITIES.len(),
        8,
        "connect/GPU/prefers-connect capability overrides"
    );
    assert_eq!(LITERATE_DOCS.len(), 5, "literate walkthrough links");
    assert_eq!(PROGRESS_NOTES.len(), 21, "long-run heads-up notes");
}

#[test]
fn progress_note_lookup_resolves_by_demo_and_line() {
    // The overfitting demo's train burst is narrated at line 9.
    let notes: Vec<_> = progress_notes_for("Watch a Model Learn (overfitting)", 9).collect();
    assert_eq!(notes.len(), 1, "exactly one note at that line");
    assert!(
        notes[0].heading.contains("six short bursts"),
        "heading body survived the toml round-trip: {:?}",
        notes[0].heading
    );
    // A line with no note returns nothing.
    assert_eq!(
        progress_notes_for("Watch a Model Learn (overfitting)", 0).count(),
        0
    );
}

#[test]
fn every_progress_note_targets_a_real_demo_line() {
    // Both invariants are pure data checks (no interpreter), so they stay in
    // the light test set: a note must name a real demo AND point at a line
    // index inside that demo's `lines`, or the browser silently drops it.
    for n in PROGRESS_NOTES {
        let demo = mlpl_web_demos::DEMOS
            .iter()
            .find(|d| d.name == n.demo)
            .unwrap_or_else(|| panic!("progress note targets unknown demo {:?}", n.demo));
        assert!(
            n.line_idx < demo.lines.len(),
            "{}: line_idx {} >= lines.len() {}",
            n.demo,
            n.line_idx,
            demo.lines.len()
        );
    }
}
