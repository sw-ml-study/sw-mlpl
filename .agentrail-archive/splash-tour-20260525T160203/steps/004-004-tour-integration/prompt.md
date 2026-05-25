Step 004: Tour integration + header button.

Wire tour into app lifecycle: show_tour UseStateHandle<bool> + tour_step UseStateHandle<usize> in UiState. Splash 'Take a guided tour' sets show_tour = true. Add 'Tour' button next to '?' in Header (reuses .help-btn CSS). Z-index layering: tour spotlight (300) above DocDialog (200). Escape closes topmost overlay.

Pages rebuild required.