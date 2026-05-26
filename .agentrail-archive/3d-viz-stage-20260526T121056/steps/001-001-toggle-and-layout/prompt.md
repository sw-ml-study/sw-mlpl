Phase 1a: 3D toggle infrastructure.

Add show_3d: UseStateHandle<bool> to UiState (default false). Add :3d REPL command (toggle, :3d on, :3d off) in a new viz3d_toggle.rs module (2 fns max). Add Ctrl+3 hotkey in the keydown handler. When show_3d is true, render a split layout: REPL output on left, <canvas id="stage3d"> on right (reuse the tutorial-split CSS pattern). The canvas is an empty black rectangle for now -- no Three.js yet.

Design to warning targets: viz3d_toggle.rs <= 4 fns, <= 20 LOC each. Pages rebuild required.