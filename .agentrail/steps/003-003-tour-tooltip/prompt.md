Step 003: Tour tooltip component.

New TourTooltip component in onboarding_tour.rs (~150 LOC, 4 functions). const TOUR_STEPS: &[TourStep] with 6 entries:

1. repl-input: 'Type expressions here. Enter to evaluate. Try 1 + 2 or iota(5).'
2. demo-select: 'Load a pre-built demo. Each runs a complete example with narration.'
3. tab-tutorial: 'Follow guided lessons, from arithmetic to transformers.'
4. tab-paths: 'Structured learning paths that sequence lessons, demos, and glossary entries.'
5. help-btn: 'Full documentation: language reference, usage guide, glossary, and diagrams.'
6. repl-input: 'Press Ctrl+Space for autocomplete. Arrow keys to navigate, Enter to accept.'

Tooltip positioning via element.get_bounding_client_rect(). CSS-only spotlight: box-shadow 0 0 0 9999px rgba(0,0,0,0.6). Back (disabled step 0), Next, Close (X). Step counter. Escape closes.

Pages rebuild required.