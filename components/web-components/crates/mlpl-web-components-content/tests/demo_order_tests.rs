//! Dropdown ordering (user direction): ML learning path first,
//! the APL2 / general-programming group after the ML groups, the
//! connect/device tiers last; names alphabetical within groups.

use mlpl_web_components_content::demo_gating::grouped_demos;

#[test]
fn ml_first_apl2_later_connect_last() {
    let groups = grouped_demos(false, &[]);
    let labels: Vec<&str> = groups.iter().map(|(l, _)| *l).collect();
    assert_eq!(labels.first(), Some(&"Basics"), "{labels:?}");
    let pos = |l: &str| labels.iter().position(|x| *x == l).unwrap_or(usize::MAX);
    assert!(
        pos("APL2 / General Programming") > pos("Training & Learning"),
        "{labels:?}"
    );
    assert!(
        pos("APL2 / General Programming") > pos("Generative Models"),
        "{labels:?}"
    );
    assert!(
        pos("CUDA - Linux GPU (connect)") > pos("APL2 / General Programming"),
        "{labels:?}"
    );
}

#[test]
fn names_alphabetical_within_groups() {
    for (label, items) in grouped_demos(false, &[]) {
        let names: Vec<String> = items
            .iter()
            .map(|(_, n, _)| n.to_ascii_lowercase())
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "section {label} not sorted");
    }
}

#[test]
fn apl2_group_has_the_eight_language_demos() {
    let groups = grouped_demos(false, &[]);
    let (_, items) = groups
        .iter()
        .find(|(l, _)| *l == "APL2 / General Programming")
        .expect("APL2 group present");
    assert_eq!(items.len(), 8, "{items:?}");
}
