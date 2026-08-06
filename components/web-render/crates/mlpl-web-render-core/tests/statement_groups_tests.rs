//! Grouping semantics for the editor Run path.

use mlpl_web_render_core::statement_groups::group_statements;

#[test]
fn multi_line_defs_stay_whole_and_singles_stay_single() {
    let src = "x = 1\n\
               def u:f(a) {\n  {data: a, size: 1}\n}\n\
               # comment\n\
               u:f(2)\n";
    let g = group_statements(src);
    assert_eq!(g.len(), 3);
    assert_eq!(g[0], "x = 1");
    assert!(g[1].starts_with("def u:f"), "{:?}", g[1]);
    assert!(g[1].ends_with('}'), "{:?}", g[1]);
    assert_eq!(g[2], "u:f(2)");
}

#[test]
fn brackets_inside_strings_and_comments_do_not_count() {
    let g = group_statements("s = \"{ not a brace\"  # neither { here\ny = 2\n");
    assert_eq!(g.len(), 2);
}

#[test]
fn trailing_unbalanced_group_is_still_emitted() {
    let g = group_statements("def u:f() {\n  1\n");
    assert_eq!(g.len(), 1);
    assert!(g[0].contains("def u:f"));
}
