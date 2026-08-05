//! Pin the Usage Guide's REPL-commands table to the runtime's
//! command registry, the same way readme_counts pins the README
//! numbers: every command in `REPL_COMMANDS` must appear in the
//! docs table, so a new command (or alias) cannot ship
//! undocumented, and the `:disp(expr)` colon-call row cannot
//! silently vanish again.

const USAGE: &str = include_str!("../../../../../docs/usage.md");

fn table_block() -> String {
    USAGE
        .lines()
        .filter(|l| l.starts_with("| "))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn every_repl_command_has_a_usage_table_row() {
    let table = table_block();
    for (name, _) in mlpl_eval::REPL_COMMANDS {
        assert!(
            table.contains(&format!(":{name}")),
            "docs/usage.md REPL table is missing :{name}"
        );
    }
}

#[test]
fn the_disp_call_form_row_is_pinned() {
    assert!(
        table_block().contains(":disp(expr)"),
        "docs/usage.md REPL table must carry the :disp(expr) row"
    );
}
