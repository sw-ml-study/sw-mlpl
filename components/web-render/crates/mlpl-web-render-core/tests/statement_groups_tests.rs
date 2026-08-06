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
    // The comment RIDES with the statement that follows it, so
    // the transcript keeps the author's narrative.
    assert_eq!(g[2], "# comment\nu:f(2)");
}

#[test]
fn comments_attach_to_the_following_statement() {
    let src = "# overview: what this document is\n\
               # APL2:  +/V\n\
               reduce(:add, v)\n\
               \n\
               # trailing orphan comment\n";
    let g = group_statements(src);
    assert_eq!(g.len(), 2, "{g:?}");
    assert_eq!(
        g[0],
        "# overview: what this document is\n# APL2:  +/V\nreduce(:add, v)"
    );
    // The trailing comment-only group survives -- it narrates
    // as a closing summary with nothing to evaluate.
    assert_eq!(g[1], "# trailing orphan comment");
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
