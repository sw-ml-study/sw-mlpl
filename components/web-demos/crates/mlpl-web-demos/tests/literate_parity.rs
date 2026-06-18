//! Regression guard against literate-doc truncation / org<->html drift.
//!
//! Root cause this prevents: commit edef03c7 truncated
//! `tictactoe-finetune.org` from 6 sections to 1 while its published
//! `.html` (manually swapped in) stayed complete -- nothing checked the
//! two stayed in sync, so the gutted `.org` shipped unnoticed.
//!
//! Invariant enforced here: for every `LITERATE_DOCS` entry, the
//! committed `.org` source (`examples/literate/`) and the committed,
//! deployed `.html` (`pages/literate/` -- note `examples/literate/*.html`
//! is gitignored/transient) expose the SAME ordered list of `mlpl` source
//! blocks (whitespace-normalized). A truncated `.org` then fails loudly:
//! its block list no longer matches the deployed html's.

use mlpl_web_demos::LITERATE_DOCS;
use std::path::PathBuf;

/// Walk up from this crate's manifest dir to the repo root (the first
/// ancestor that contains `examples/literate`).
fn repo_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !dir.join("examples/literate").is_dir() {
        assert!(dir.pop(), "repo root (with examples/literate) not found");
    }
    dir
}

/// Collapse all whitespace runs to single spaces and trim, so syntax
/// highlighting / indentation differences don't cause false mismatches.
fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The ordered, normalized `mlpl` source blocks declared in an `.org`.
fn org_blocks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(l) = lines.next() {
        if !l.trim_start().starts_with("#+begin_src mlpl") {
            continue;
        }
        let body: Vec<&str> = lines
            .by_ref()
            .take_while(|x| !x.trim_start().starts_with("#+end_src"))
            .collect();
        out.push(norm(&body.join(" ")));
    }
    out
}

/// Strip `<...>` tags, then unescape the entities org-export emits.
fn untag(seg: &str) -> String {
    let mut s = String::new();
    let mut in_tag = false;
    for c in seg.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => s.push(c),
            _ => {}
        }
    }
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#34;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// The ordered, normalized `mlpl` source blocks rendered in an `.html`.
fn html_blocks(text: &str) -> Vec<String> {
    let open = "<pre class=\"src src-mlpl\">";
    text.split(open)
        .skip(1)
        .filter_map(|seg| seg.split_once("</pre>"))
        .map(|(block, _)| norm(&untag(block)))
        .collect()
}

#[test]
fn every_literate_doc_org_matches_its_published_html() {
    let lit = repo_root().join("examples/literate");
    let pages = repo_root().join("pages/literate");
    for (demo, html_name) in LITERATE_DOCS {
        let stem = html_name.strip_suffix(".html").expect("html name");
        let org = lit.join(format!("{stem}.org"));
        // The committed, deployed html lives in pages/literate/
        // (examples/literate/*.html is gitignored/transient).
        let html = pages.join(html_name);
        assert!(org.is_file(), "{demo}: missing source {org:?}");
        assert!(
            html.is_file(),
            "{demo}: {html_name} not tracked/deployed in pages/literate (run build-pages.sh)"
        );
        let ob = org_blocks(&std::fs::read_to_string(&org).unwrap());
        let hb = html_blocks(&std::fs::read_to_string(&html).unwrap());
        assert!(!ob.is_empty(), "{demo}: {org:?} has no mlpl source blocks");
        assert_eq!(
            ob,
            hb,
            "{demo}: {stem}.org has {} mlpl blocks but {stem}.html has {} -- \
             org<->html drift (truncation?). Re-publish via publish.sh.",
            ob.len(),
            hb.len()
        );
    }
}

/// Every `.org` in `examples/literate/` must have a committed, deployed
/// `.html` in `pages/literate/` (no source whose output isn't shipped).
#[test]
fn no_orphan_literate_org() {
    let lit = repo_root().join("examples/literate");
    let pages = repo_root().join("pages/literate");
    for entry in std::fs::read_dir(&lit).unwrap().flatten() {
        let p = entry.path();
        if p.extension().is_some_and(|e| e == "org") {
            let html = format!("{}.html", p.file_stem().unwrap().to_str().unwrap());
            assert!(
                pages.join(&html).is_file(),
                "{p:?} has no deployed pages/literate/{html} (run build-pages.sh)"
            );
        }
    }
}
