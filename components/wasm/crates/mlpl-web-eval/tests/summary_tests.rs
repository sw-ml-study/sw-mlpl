//! Saga 32 step 004: tests for `summary::summarize`,
//! moved out of `summary.rs`'s inline `#[cfg(test)] mod
//! tests` block so the production module stays under the
//! sw-checklist function-count budget.

use mlpl_web_eval::summary::summarize;

#[test]
fn short_output_is_not_summarized() {
    assert!(summarize("42").is_none());
    assert!(summarize("1 2 3").is_none());
    assert!(summarize("1 2 3\n4 5 6").is_none());
}

#[test]
fn long_vector_is_summarized_as_1d() {
    let out = (0..20).map(|i| i.to_string()).collect::<Vec<_>>().join(" ");
    // long enough via chars? 20 numbers fit under 200 chars, so pad.
    let out = format!("{out}\n{out}\n{out}\n{out}\n{out}\n{out}\n{out}\n{out}\n{out}");
    let s = summarize(&out).expect("should summarize");
    assert_eq!(s.min, 0.0);
    assert_eq!(s.max, 19.0);
}

#[test]
fn uniform_matrix_reports_rows_cols() {
    let out = (0..9)
        .map(|i| format!("{i} {i} {i} {i} {i} {i} {i} {i} {i} {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let s = summarize(&out).expect("should summarize");
    assert!(s.shape.contains("9x10"), "shape was {}", s.shape);
}

#[test]
fn non_numeric_output_is_not_summarized() {
    let out = "line one\nline two\nline three\nline four\nline five\n\
               line six\nline seven\nline eight\nline nine\nline ten";
    assert!(summarize(out).is_none());
}

#[test]
fn stats_are_correct_for_known_vector() {
    // 1..=20, long enough to trigger via line count.
    let rows: Vec<String> = (1..=20).map(|i| i.to_string()).collect();
    let out = rows.join("\n");
    let s = summarize(&out).expect("should summarize");
    assert_eq!(s.min, 1.0);
    assert_eq!(s.max, 20.0);
    assert!((s.mean - 10.5).abs() < 1e-9);
    assert!((s.median - 10.5).abs() < 1e-9);
}
