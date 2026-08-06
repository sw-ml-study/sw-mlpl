//! Narration/code splitting for editor Run groups.

use mlpl_web_eval_core::state::split_leading_comments;

#[test]
fn leading_comments_become_stripped_narration() {
    let (n, c) = split_leading_comments("# What you are about to see\n#   indented note\nx = 1");
    assert_eq!(
        n.as_deref(),
        Some("What you are about to see\nindented note")
    );
    assert_eq!(c.as_deref(), Some("x = 1"));
}

#[test]
fn comment_only_group_is_pure_narration() {
    let (n, c) = split_leading_comments("# What this showed\n# a summary");
    assert_eq!(n.as_deref(), Some("What this showed\na summary"));
    assert!(c.is_none());
}

#[test]
fn plain_code_has_no_narration() {
    let (n, c) = split_leading_comments("y = 2");
    assert!(n.is_none());
    assert_eq!(c.as_deref(), Some("y = 2"));
}

#[test]
fn interior_comments_stay_inside_the_code() {
    let src = "# heading\ndef u:f() {\n  # interior note\n  1\n}";
    let (n, c) = split_leading_comments(src);
    assert_eq!(n.as_deref(), Some("heading"));
    let code = c.unwrap();
    assert!(code.contains("# interior note"), "{code}");
    assert!(code.starts_with("def u:f"), "{code}");
}
