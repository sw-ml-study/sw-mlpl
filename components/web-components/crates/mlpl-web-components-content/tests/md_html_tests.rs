//! The build-time markdown -> HTML converter (doc tabs): tables
//! become real <table> rows, code fences stay literal, inline
//! backticks become <code>, and text is HTML-escaped.

#[allow(dead_code)] // the harness view only exercises convert()
#[path = "../build_md_html.rs"]
mod build_md_html;

use build_md_html::convert;

#[test]
fn pipe_table_becomes_html_table() {
    let md = "| A | B |\n| --- | --- |\n| `x` | 1 < 2 |\n";
    let h = convert(md);
    assert!(h.contains("<table><tr><th>A</th><th>B</th></tr>"), "{h}");
    assert!(
        h.contains("<td><code>x</code></td><td>1 &lt; 2</td>"),
        "{h}"
    );
    assert!(!h.contains("| ---"), "separator row dropped: {h}");
}

#[test]
fn code_fence_is_escaped_verbatim() {
    let md = "```\nif a < b { }\n```\n";
    let h = convert(md);
    assert!(
        h.contains("<pre><code>if a &lt; b { }\n</code></pre>"),
        "{h}"
    );
}

#[test]
fn headings_paragraphs_and_lists() {
    let md = "## Title\n\nSome `inline` text\nacross two lines.\n\n- first\n- second\n";
    let h = convert(md);
    assert!(h.contains("<h2>Title</h2>"), "{h}");
    assert!(
        h.contains("<p>Some <code>inline</code> text across two lines.</p>"),
        "{h}"
    );
    assert!(h.contains("<ul><li>first</li><li>second</li></ul>"), "{h}");
}
