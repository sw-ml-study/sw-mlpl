//! The loader against the in-memory provider: expansion order,
//! load-once, relative nesting, cycle chains, sandbox rules, and
//! parse-error attribution (upstream tests 2-7 of the contract).

use mlpl_source_loader::{IncludeError, MemoryProvider, SourceId, expand};

fn ids(chunks: &[mlpl_source_loader::Chunk]) -> Vec<&str> {
    chunks.iter().map(|c| c.source.0.as_str()).collect()
}

#[test]
fn definitions_splice_at_the_include_site_in_source_order() {
    let p = MemoryProvider::default()
        .with("root.mlpl", "a = 1\ninclude \"lib.mlpl\"\nc = 3\n")
        .with("lib.mlpl", "b = 2\n");
    let (chunks, table) = expand(&SourceId("root.mlpl".into()), &p).unwrap();
    assert_eq!(ids(&chunks), ["root.mlpl", "lib.mlpl", "root.mlpl"]);
    assert_eq!(chunks[0].stmts.len(), 1);
    assert_eq!(chunks[1].stmts.len(), 1);
    assert_eq!(chunks[2].stmts.len(), 1);
    assert!(
        table
            .text(&SourceId("lib.mlpl".into()))
            .unwrap()
            .contains("b = 2")
    );
}

#[test]
fn nested_includes_resolve_relative_to_the_including_file() {
    let p = MemoryProvider::default()
        .with("root.mlpl", "include \"sub/mid.mlpl\"\n")
        .with("sub/mid.mlpl", "include \"leaf.mlpl\"\nm = 1\n")
        .with("sub/leaf.mlpl", "l = 2\n");
    let (chunks, _) = expand(&SourceId("root.mlpl".into()), &p).unwrap();
    assert_eq!(ids(&chunks), ["sub/leaf.mlpl", "sub/mid.mlpl"]);
}

#[test]
fn duplicate_includes_load_once() {
    let p = MemoryProvider::default()
        .with(
            "root.mlpl",
            "include \"a.mlpl\"\ninclude \"a.mlpl\"\nx = 1\n",
        )
        .with("a.mlpl", "a = 1\n");
    let (chunks, _) = expand(&SourceId("root.mlpl".into()), &p).unwrap();
    assert_eq!(ids(&chunks), ["a.mlpl", "root.mlpl"]);
}

#[test]
fn cycles_report_the_complete_chain() {
    let p = MemoryProvider::default()
        .with("a.mlpl", "include \"b.mlpl\"\n")
        .with("b.mlpl", "include \"c.mlpl\"\n")
        .with("c.mlpl", "include \"a.mlpl\"\n");
    let err = expand(&SourceId("a.mlpl".into()), &p).unwrap_err();
    let IncludeError::Cycle { chain } = err else {
        panic!("expected cycle, got {err:?}")
    };
    assert_eq!(chain, ["a.mlpl", "b.mlpl", "c.mlpl", "a.mlpl"]);
}

#[test]
fn direct_self_include_is_a_cycle() {
    let p = MemoryProvider::default().with("a.mlpl", "include \"a.mlpl\"\n");
    let err = expand(&SourceId("a.mlpl".into()), &p).unwrap_err();
    assert!(matches!(err, IncludeError::Cycle { .. }), "{err:?}");
}

#[test]
fn sandbox_rejects_absolute_and_escaping_paths() {
    let p = MemoryProvider::default()
        .with("root.mlpl", "include \"/etc/x.mlpl\"\n")
        .with("esc.mlpl", "include \"../outside.mlpl\"\n");
    let err = expand(&SourceId("root.mlpl".into()), &p).unwrap_err();
    assert!(err.to_string().contains("absolute"), "{err}");
    let err = expand(&SourceId("esc.mlpl".into()), &p).unwrap_err();
    assert!(err.to_string().contains("escapes the source root"), "{err}");
}

#[test]
fn parse_errors_name_the_failing_file() {
    let p = MemoryProvider::default()
        .with("root.mlpl", "include \"bad.mlpl\"\n")
        .with("bad.mlpl", "x = = 1\n");
    let err = expand(&SourceId("root.mlpl".into()), &p).unwrap_err();
    let IncludeError::Parse { source, .. } = err else {
        panic!("expected parse error, got {err:?}")
    };
    assert_eq!(source, "bad.mlpl");
}

#[test]
fn missing_file_is_a_structured_error() {
    let p = MemoryProvider::default().with("root.mlpl", "include \"gone.mlpl\"\n");
    let err = expand(&SourceId("root.mlpl".into()), &p).unwrap_err();
    assert!(err.to_string().contains("no such source file"), "{err}");
}
