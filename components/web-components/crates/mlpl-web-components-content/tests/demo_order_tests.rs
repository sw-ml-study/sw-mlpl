//! Dropdown ordering (user direction): ML learning path first,
//! the APL2 / general-programming group after the ML groups, the
//! connect/device tiers last; names alphabetical within groups.

use mlpl_web_components_content::demo_gating::{GROUP_TOOLTIPS, demo_tooltip, grouped_demos};

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
    // Experiment Quality sits between Training & Learning and the
    // connect tiers (curated placement, not the alphabetical
    // fallback bucket).
    assert!(
        pos("Experiment Quality") > pos("Training & Learning")
            && pos("Experiment Quality") < pos("CUDA - Linux GPU (connect)"),
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
fn every_group_and_demo_has_explanatory_tooltip_text() {
    for (group, demos) in grouped_demos(false, &[]) {
        let group_tip = GROUP_TOOLTIPS.iter().find(|(name, _)| *name == group);
        assert!(group_tip.is_some(), "missing tooltip for {group}");
        for (index, name, disabled) in demos {
            let tip = demo_tooltip(index, group, disabled);
            assert!(
                tip.contains(name),
                "tooltip does not identify {name}: {tip}"
            );
            assert!(
                tip.len() > name.len() + 10,
                "tooltip does not explain {name}"
            );
        }
    }
}

#[test]
fn disabled_demo_tooltip_explains_its_connection_requirement() {
    let groups = grouped_demos(false, &[]);
    for (group, demos) in groups {
        for (index, _, disabled) in demos.into_iter().filter(|(_, _, disabled)| *disabled) {
            let tip = demo_tooltip(index, group, disabled);
            assert!(
                tip.contains("Unavailable:"),
                "missing disabled reason: {tip}"
            );
            assert!(
                tip.contains("connect"),
                "missing connection guidance: {tip}"
            );
        }
    }
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
fn apl2_group_has_the_language_demos() {
    let groups = grouped_demos(false, &[]);
    let (_, items) = groups
        .iter()
        .find(|(l, _)| *l == "APL2 / General Programming")
        .expect("APL2 group present");
    assert_eq!(items.len(), 12, "{items:?}");
}
