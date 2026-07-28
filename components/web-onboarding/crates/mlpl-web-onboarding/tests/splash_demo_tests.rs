//! The splash "Run the Basics demo" card must target the demo NAMED
//! Basics -- a raw index rotted when demos.toml gained/reordered
//! entries (observed: index 7 ran Game of Life).

use mlpl_web_onboarding::splash::basics_demo_index;

#[test]
fn splash_basics_card_targets_the_basics_demo() {
    let idx = basics_demo_index();
    assert_eq!(
        mlpl_web_demos::DEMOS[idx].name,
        "Basics",
        "splash card must resolve the Basics demo by name"
    );
}
