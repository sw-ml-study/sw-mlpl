//! `split_group_lines` -- per-line comment splitting for an entry
//! (statement group). Regression for upstream-asks #16: a leading
//! full-line `# comment` must NOT swallow the code on the lines after it.

use mlpl_web_tutorial::{split_group_lines, split_inline_comment};

#[test]
fn a_leading_comment_line_does_not_swallow_the_following_code() {
    let group = "# build the operation table\nt = table(:u:fight, range(3), range(3))";
    let lines = split_group_lines(group);
    assert_eq!(lines.len(), 2);
    // The comment line: no code, the comment text.
    assert_eq!(lines[0], ("", Some("build the operation table")));
    // The statement line renders as CODE, not inside the comment span.
    assert_eq!(lines[1], ("t = table(:u:fight, range(3), range(3))", None));
}

#[test]
fn single_line_groups_are_unchanged() {
    // A one-line entry (the built-in catalog case) splits exactly as
    // split_inline_comment does -- one pair, same result.
    let line = "x = 1  # a note";
    assert_eq!(split_group_lines(line), vec![split_inline_comment(line)]);
    assert_eq!(split_group_lines(line), vec![("x = 1", Some("a note"))]);
}

#[test]
fn an_inline_comment_on_a_code_line_still_splits() {
    let group = "t = f(x)  # inline\n# trailing whole-line note";
    let lines = split_group_lines(group);
    assert_eq!(lines[0], ("t = f(x)", Some("inline")));
    assert_eq!(lines[1], ("", Some("trailing whole-line note")));
}

#[test]
fn an_empty_group_yields_one_empty_line() {
    // So the prompt still renders for an empty input.
    assert_eq!(split_group_lines(""), vec![("", None)]);
}
