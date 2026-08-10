# Saga: web-demo-picker-splash

Two web-UI changes sharing one pages rebuild + deploy:

1. Replace the native <select><optgroup> demo picker with a
   CUSTOM, fully-stylable dropdown so demo-group NAMES are legible
   in every browser (Safari ignores <optgroup> styling; Chrome
   ignores its font-size -- the native element cannot be styled
   reliably). New DemoDropdown Yew component: a toggle button + a
   panel of styled group headers (14px, 700, --lavender) and demo
   rows. MUST preserve: capability gating (disabled + hint via
   demo_gating), alphabetical order within groups + SECTION_ORDER,
   the "Load Demo..." placeholder, aria-label, the tour target
   (data-tour-target="demo-select"), tutorial_active hiding, and
   the on_demo(usize) callback. Live/CPU demos ungated.

2. Show the build's UTC Zulu timestamp on the splash so a viewer
   can identify the build regardless of local timezone. The value
   already exists: mlpl-web build_env.rs emits BUILD_TIMESTAMP via
   `date -u +%Y-%m-%dT%H:%M:%SZ`. Thread it into SplashOverlay (new
   build_time prop) and render it under the version label.

3. Rebuild pages + deploy (CPU/live demo, no serve rebuild). Close.

## Steps
1. custom-demo-dropdown -- new demo_dropdown.rs (DemoDropdown) in
   mlpl-web-components-content; mode_bar wires it in place of the
   <select>; CSS for .demo-dropdown / panel / group label / item in
   index.html. Keep demo_order_tests green; clippy/fmt.
2. splash-build-timestamp -- BUILD_TIMESTAMP -> SplashProps
   build_time -> rendered line; splash CSS; splash test if present.
3. rebuild-deploy -- build-pages.sh, commit pages/, deploy-pages.sh,
   ts-suffixed review URL, --done.
