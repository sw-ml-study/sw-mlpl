Step 005: What's new modal.

New WhatsNewOverlay in onboarding_whats_new.rs (~100 LOC, 3 functions). Shows when should_show_whats_new() returns true (stored version < compiled version).

Content: const WHATS_NEW_ITEMS with heading + body pairs. Initial items: Guided tour, REPL autocomplete, Dimensionality reduction. 'Got it' dismiss writes current version to localStorage. Splash takes priority over what's-new on first visits.

Pages rebuild required.