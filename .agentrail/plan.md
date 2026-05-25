# Splash Screen + Guided Tour Milestone

Saga 34, proposed.

## Why this exists

The MLPL web playground ships 31 demos, 32+ tutorial lessons,
a glossary, learning paths, a completion popup, and a
documentation dialog -- but a first-time visitor sees only
three lines of grey text ("Welcome to MLPL. Type expressions
and press Enter.") and an empty REPL. There is no onboarding,
no feature discovery, and no "what's new" announcement
surface. Users who arrive via a link or search engine must
self-discover every feature by clicking around.

This milestone adds three coordinated overlays:

1. **Splash/onboarding modal** -- shown on first visit,
   dismissable, re-showable from a "Tour" button. Uses
   `images/splash-bg.png` as the background image. Tells
   the user what MLPL is and offers clickable quick-start
   actions.
2. **Guided tour** -- a step-by-step tooltip walkthrough
   that highlights each major UI element and explains its
   purpose in one sentence. Navigable with Back/Next/Escape.
3. **"What's new" modal** -- shown once per version bump,
   listing recent feature additions. Static content updated
   each saga.

Together these three surfaces ensure that every user --
first-timer, returner after a version bump, or someone who
just forgot what a button does -- can orient within 30
seconds.

## Non-goals

- **Third-party tour library.** The app is pure Yew (Rust
  WASM) with inline CSS. Adding a JS tour library (Shepherd,
  Intro.js) would require JS interop glue and a CDN dep. The
  tour is simple enough (6 steps) to build natively in Yew.
- **Animated transitions.** Fade-in/out on the spotlight is
  nice-to-have but not blocking.
- **Persistent tour progress.** The tour is short (6 steps).
  If the user exits mid-tour and re-opens, it restarts from
  step 1.
- **Internationalization.** All copy is English, hardcoded as
  `&'static str` constants.
- **Mobile-specific layout.** Tour tooltips use the same
  responsive flow as the rest of the app but no
  mobile-specific positioning logic is added.

## Dependencies

No hard blockers. All required infrastructure (modal pattern,
Escape handling, element IDs, Yew component conventions)
already exists. `web-sys` features `Storage` and `DomRect`
must be added to `apps/mlpl-web/Cargo.toml` (compile-time
feature flags, no runtime cost).

## What already exists

- **`Welcome` component** (`components.rs`): three lines of
  greeting text. Superseded by the splash but stays as the
  post-dismissal in-REPL greeting.
- **`DocDialog` component** (`components.rs`): full
  modal-backdrop + modal-body pattern with tabs, close button,
  Escape-to-close. Splash and what's-new reuse this CSS
  pattern.
- **`GlossaryPopupHost`** (`glossary_popup.rs`): window-level
  overlay via `use_effect_with` + `Closure::forget`. Tour
  tooltip follows the same architecture.
- **Element IDs:** `#repl-input`, `#output`, plus `aria-label`
  attributes on the demo dropdown, help button, completion
  popup. Three more `data-tour-target` attributes needed.
- **Demo runner:** `handlers::make_run_demo` fires a demo by
  index. Splash's "Try these" buttons reuse
  `on_demo.emit(idx)`.
- **Version info:** `BUILD_SHA` + `BUILD_TIMESTAMP` from
  `build_env.rs`; workspace version `0.20.0` in root
  `Cargo.toml`.
- **Background image:** `images/splash-bg.png` (terminal
  screenshot, dark theme) -- specified by user for the splash
  overlay background.
- **UI screenshots:** `images/01-repl.png` through
  `images/06-glossary.png` -- available for tour step
  illustrations if needed.

## Quality requirements (every step)

Identical to saga 33. TDD; four `cargo` gates (test, clippy,
fmt, doc) + `markdown-checker` + `sw-checklist` green;
`/mw-cp` checkpoint; push after every commit; web changes
rebuild `pages/`; `.agentrail/` committed.

New modules must stay under the project's 5-fn/module,
500-line/file budgets.

sw-checklist ratchet: each commit must strictly lower BOTH
the failed count AND the warnings count.

## Steps

### Step 001 -- localStorage helper + data-tour-target IDs

New module `onboarding_storage.rs` (~60 LOC, 3 functions)
wrapping `web_sys::Storage` for two keys:

- `mlpl_splash_dismissed: bool` -- true after dismissal
- `mlpl_last_seen_version: String` -- version last shown in
  what's-new

Pure predicates `should_show_splash(dismissed)` and
`should_show_whats_new(last_seen, current)`.

Add `data-tour-target` attributes to six target elements
in `components.rs` and `render_shell_header.rs`:
`repl-input`, `demo-select`, `tab-tutorial`, `tab-paths`,
`help-btn`, `completion-popup`.

Add `"Storage"` and `"DomRect"` to `web-sys` features in
`Cargo.toml`.

Tests: unit tests for both pure predicates.

### Step 002 -- Splash/onboarding overlay

New `SplashOverlay` component in `onboarding_splash.rs`
(~120 LOC, 3 functions). Uses `images/splash-bg.png` as
CSS `background-image` on the backdrop with a dark overlay.

Contents:
- Headline: "Welcome to sw-MLPL" with badge image
- Subtitle: "An array programming language for learning
  machine learning, from scalars to transformers."
- Four clickable quick-start cards:
  1. "Run the Basics demo"
  2. "Try `1 + 2` in the REPL"
  3. "Open a tutorial lesson"
  4. "Explore learning paths"
- "Dismiss" button (localStorage, closes overlay)
- "Take a guided tour" button (closes overlay, fires tour)

Integration: rendered conditionally in `render_shell.rs`
(or new `render_shell_overlays.rs`). New
`show_splash: UseStateHandle<bool>` in UiState.

### Step 003 -- Tour tooltip component

New `TourTooltip` component in `onboarding_tour.rs` (~150
LOC, 4 functions).

`const TOUR_STEPS: &[TourStep]` (6 entries):

1. `repl-input` -- "Type expressions here. Enter to
   evaluate. Try `1 + 2` or `iota(5)`."
2. `demo-select` -- "Load a pre-built demo. Each runs a
   complete example with narration."
3. `tab-tutorial` -- "Follow guided lessons, from
   arithmetic to transformers."
4. `tab-paths` -- "Structured learning paths that sequence
   lessons, demos, and glossary entries."
5. `help-btn` -- "Full documentation: language reference,
   usage guide, glossary, and diagrams."
6. `repl-input` -- "Press Ctrl+Space for autocomplete.
   Arrow keys to navigate, Enter to accept."

Tooltip positioning via `element.get_bounding_client_rect()`.
CSS-only spotlight: `box-shadow: 0 0 0 9999px rgba(0,0,0,0.6)`
on a positioned overlay around the target.

Back (disabled on step 0), Next, Close (X). Step counter:
"Step 1 of 6". Escape closes.

### Step 004 -- Tour integration + header button

Wire tour into app lifecycle:
- `show_tour: UseStateHandle<bool>` +
  `tour_step: UseStateHandle<usize>` in UiState
- Splash "Take a guided tour" sets `show_tour = true`
- Add "Tour" button next to "?" in Header (reuses
  `.help-btn` CSS)
- Z-index layering: tour spotlight (300) above DocDialog
  (200)
- Escape closes topmost overlay

### Step 005 -- "What's new" modal

New `WhatsNewOverlay` in `onboarding_whats_new.rs` (~100
LOC, 3 functions). Shows when `should_show_whats_new()`
returns true (stored version < compiled version).

Content: `const WHATS_NEW_ITEMS: &[(&str, &str)]` --
heading + body pairs. Initial items:
- "Guided tour" -- "Click Tour to walk through every feature."
- "REPL autocomplete" -- "Press Ctrl+Space for suggestions."
- "Dimensionality reduction" -- "UMAP, MDS, random projection,
  critical-dimensions heatmaps."

"Got it" dismiss writes current version to localStorage.

Splash takes priority over what's-new on first visits.

### Step 006 -- Overlay lifecycle + pages rebuild

Extract overlay rendering into `render_shell_overlays.rs`
if not already done. Single place for overlay priority and
z-index stacking:

Lifecycle rules:
- First-ever visit: splash shows
- Splash "Tour" -> splash closes, tour starts
- Splash "Dismiss" -> splash closes, nothing else
- Return visit same version: nothing automatic
- Return visit new version: what's-new shows
- "Tour" button in header: tour starts unconditionally
- Escape closes topmost overlay

Final CSS polish: Catppuccin Mocha colors, responsive splash
cards, tour tooltip arrow, keyboard accessibility,
aria- attributes.

Rebuild `pages/` and commit.

## Architecture notes

### Module budget compliance

| New file | Est. LOC | Est. fns | Budget |
|---|---|---|---|
| `onboarding_storage.rs` | ~60 | 3 | OK |
| `onboarding_splash.rs` | ~120 | 3 | OK |
| `onboarding_tour.rs` | ~150 | 4 | OK |
| `onboarding_whats_new.rs` | ~100 | 3 | OK |
| `render_shell_overlays.rs` | ~60 | 2 | OK |

Total new Rust: ~490 LOC across 5 files.

### Element targeting

`data-tour-target` custom attributes (greppable, no ID
namespace collision). Tour queries
`[data-tour-target='name']`.

### localStorage keys

Two keys, `mlpl_`-prefixed:
- `mlpl_splash_dismissed` -- value `"1"` or absent
- `mlpl_last_seen_version` -- semver string or absent

Fail-open: absent/unparseable key shows the overlay
(correct for onboarding).

### Tour tooltip positioning

1. `element.get_bounding_client_rect()` for target's
   viewport-relative box
2. Each `TourStep` declares preferred `TooltipPosition`
   (Below for REPL input, Above for header tabs)
3. Fallback to opposite side if tooltip would clip
4. `position: fixed` with computed `top`/`left` inline
   styles
