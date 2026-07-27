//! Minimal markdown -> HTML converter for the doc tabs: exactly
//! the subset those two files use -- ATX headings, fenced code
//! blocks, pipe tables (the point: real <table> so columns line
//! up), bullet lists, paragraphs, and inline `code`. Everything
//! is HTML-escaped; the output carries a leading inline <style>
//! scoped under .doc-html.

use std::env;
use std::fs;
use std::path::Path;

const STYLE: &str = "<style>\
.doc-html table { border-collapse: collapse; margin: 10px 0; width: 100%; }\
.doc-html th, .doc-html td { border: 1px solid var(--surface1, #45475a); \
padding: 4px 8px; text-align: left; vertical-align: top; }\
.doc-html th { background: var(--surface0, #313244); }\
.doc-html tr:nth-child(even) td { background: rgba(127, 127, 127, 0.06); }\
.doc-html pre { background: var(--mantle, #181825); padding: 8px 10px; \
border-radius: 6px; overflow-x: auto; }\
.doc-html code { font-family: inherit; background: rgba(127, 127, 127, 0.15); \
padding: 0 3px; border-radius: 3px; }\
.doc-html pre code { background: none; padding: 0; }\
.doc-html h1, .doc-html h2, .doc-html h3 { margin: 14px 0 6px; }\
.doc-html p, .doc-html li { line-height: 1.5; }\
</style>";

pub fn emit(md_path: &str, out_name: &str) {
    let md = fs::read_to_string(md_path).unwrap_or_else(|e| panic!("read {md_path}: {e}"));
    let html = format!("<div class=\"doc-html\">{STYLE}{}</div>", convert(&md));
    let out = Path::new(&env::var("OUT_DIR").unwrap()).join(out_name);
    fs::write(out, html).expect("write doc html");
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Escape, then re-introduce inline code for `...` spans.
fn inline(s: &str) -> String {
    let e = esc(s);
    let mut out = String::new();
    let mut in_code = false;
    for part in e.split('`') {
        if in_code {
            out.push_str("<code>");
            out.push_str(part);
            out.push_str("</code>");
        } else {
            out.push_str(part);
        }
        in_code = !in_code;
    }
    out
}

fn table_row(line: &str, tag: &str) -> String {
    let cells: Vec<&str> = line.trim().trim_matches('|').split('|').collect();
    let tds: String = cells
        .iter()
        .map(|c| format!("<{tag}>{}</{tag}>", inline(c.trim())))
        .collect();
    format!("<tr>{tds}</tr>")
}

fn is_separator_row(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|') && t.chars().all(|c| "|-: ".contains(c))
}

pub fn convert(md: &str) -> String {
    let mut out = String::new();
    let mut lines = md.lines().peekable();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("```") {
            let _ = rest;
            out.push_str("<pre><code>");
            for l in lines.by_ref() {
                if l.starts_with("```") {
                    break;
                }
                out.push_str(&esc(l));
                out.push('\n');
            }
            out.push_str("</code></pre>");
        } else if let Some(h) = heading(line) {
            out.push_str(&h);
        } else if line.trim_start().starts_with('|') {
            out.push_str("<table>");
            out.push_str(&table_row(line, "th"));
            while let Some(&next) = lines.peek() {
                if !next.trim_start().starts_with('|') {
                    break;
                }
                let row = lines.next().unwrap();
                if !is_separator_row(row) {
                    out.push_str(&table_row(row, "td"));
                }
            }
            out.push_str("</table>");
        } else if let Some(item) = line.strip_prefix("- ") {
            out.push_str("<ul>");
            out.push_str(&format!("<li>{}</li>", inline(item)));
            while let Some(&next) = lines.peek() {
                if let Some(it) = next.strip_prefix("- ") {
                    out.push_str(&format!("<li>{}</li>", inline(it)));
                    lines.next();
                } else if next.starts_with("  ") && !next.trim().is_empty() {
                    // continuation line of the previous bullet
                    out.push_str(&format!(" {}", inline(next.trim())));
                    lines.next();
                } else {
                    break;
                }
            }
            out.push_str("</ul>");
        } else if line.trim().is_empty() {
            // paragraph break; nothing to emit
        } else {
            // paragraph: gather until blank/structural line
            let mut para = inline(line);
            while let Some(&next) = lines.peek() {
                let structural = next.trim().is_empty()
                    || next.starts_with('#')
                    || next.starts_with("```")
                    || next.starts_with("- ")
                    || next.trim_start().starts_with('|');
                if structural {
                    break;
                }
                para.push(' ');
                para.push_str(&inline(next));
                lines.next();
            }
            out.push_str(&format!("<p>{para}</p>"));
        }
    }
    out
}

fn heading(line: &str) -> Option<String> {
    for (hashes, tag) in [("#### ", "h4"), ("### ", "h3"), ("## ", "h2"), ("# ", "h1")] {
        if let Some(text) = line.strip_prefix(hashes) {
            return Some(format!("<{tag}>{}</{tag}>", inline(text)));
        }
    }
    None
}
