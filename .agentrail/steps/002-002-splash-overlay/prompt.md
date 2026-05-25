Step 002: Splash/onboarding overlay.

New SplashOverlay component in onboarding_splash.rs (~120 LOC, 3 functions). Uses images/splash-bg.png as CSS background-image on the backdrop with a dark overlay.

Contents: headline 'Welcome to sw-MLPL' with badge, subtitle, four clickable quick-start cards (Run Basics demo, Try 1+2 in REPL, Open tutorial, Explore paths), Dismiss button (localStorage), Take a guided tour button.

Integration: rendered conditionally in render_shell.rs (or new render_shell_overlays.rs). New show_splash UseStateHandle<bool> in UiState initialized from localStorage.

Copy splash-bg.png into the apps/mlpl-web asset pipeline (trunk serves from the app dir, so the image needs to be accessible to the WASM build).

Pages rebuild required.