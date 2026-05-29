//! Saga 33 step 042 path-resume tests, lifted from the inline
//! `#[cfg(test)] mod tests` block in render_modes.rs into
//! integration tests during the saga 82 step 8 carve-out.
//!
//! The key invariant: an in-progress `cur_path` survives across
//! lesson navigation, but the WALKER pane only renders when no
//! lesson is open -- so the lesson pane wins the screen while
//! the walker position is held in state for the "Back to path"
//! button to resume.

use mlpl_web_components_chrome::header::HeaderMode;
use mlpl_web_render_core::modes::compute_modes;

#[test]
fn cold_start_is_repl_mode() {
    let m = compute_modes(None, None, false);
    assert!(!m.tutorial_active);
    assert!(!m.paths_active);
    assert_eq!(m.header_mode, HeaderMode::Repl);
}

#[test]
fn lesson_open_alone_is_tutorial_mode() {
    let m = compute_modes(Some(3), None, false);
    assert!(m.tutorial_active);
    assert!(!m.paths_active);
    assert_eq!(m.header_mode, HeaderMode::Tutorial);
}

#[test]
fn path_open_alone_is_paths_mode() {
    let m = compute_modes(None, Some((Some(0), 2)), false);
    assert!(!m.tutorial_active);
    assert!(m.paths_active);
    assert_eq!(m.header_mode, HeaderMode::Paths);
}

#[test]
fn lesson_open_with_path_in_progress_renders_lesson_not_walker() {
    // The path-resume bug fix: when both states are set (user
    // navigated to a lesson FROM a path walker), the lesson
    // pane wins. `cur_path` is preserved for "Back to path"
    // restoration.
    let m = compute_modes(Some(5), Some((Some(0), 2)), false);
    assert!(m.tutorial_active);
    assert!(
        !m.paths_active,
        "paths walker must NOT render while a lesson is open"
    );
    assert!(
        m.cur_path.is_some(),
        "path position must be preserved for resume"
    );
    assert_eq!(m.header_mode, HeaderMode::Tutorial);
}
