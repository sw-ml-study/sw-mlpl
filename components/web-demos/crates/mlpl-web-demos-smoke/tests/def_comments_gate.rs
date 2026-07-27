//! Enforcement gate (user directive, demo-comments-complete saga):
//! NO demo `def u:` may have zero comments. Every definition must
//! open with a doc-string (leading string literal), and every def
//! line must lex + parse -- including the ones inside demos the
//! quick registry sweep skips as heavy.

use mlpl_parser::{lex, parse};
use mlpl_web_demos::DEMOS;

#[test]
fn every_demo_def_has_a_doc_string_and_parses() {
    let mut checked = 0;
    for demo in DEMOS.iter() {
        for (i, line) in demo.lines.iter().enumerate() {
            let t = line.trim_start();
            if !t.starts_with("def u:") {
                continue;
            }
            checked += 1;
            let brace = t
                .find('{')
                .unwrap_or_else(|| panic!("[{} line {i}] def without a body: {t}", demo.name));
            let after = t[brace + 1..].trim_start();
            assert!(
                after.starts_with('"'),
                "[{} line {i}] def u: without a leading doc-string: {t}",
                demo.name
            );
            let toks = lex(line).unwrap_or_else(|e| panic!("[{} line {i}] lex: {e:?}", demo.name));
            parse(&toks).unwrap_or_else(|e| panic!("[{} line {i}] parse: {e:?}", demo.name));
        }
    }
    assert!(checked >= 25, "expected to check many defs, saw {checked}");
}
