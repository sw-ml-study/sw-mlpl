# 3D Sculpture Dimension Orientation (normative)

This is the canonical rule for how a tensor's axes map to the 3D
scene's axes in the web playground's 3D view. It is a recurring
design requirement; treat it as normative for every sculpture
renderer in `components/web/crates/mlpl-web/js/stage3d.js`.

## The rule

Order a tensor's axes longest-first and assign them to scene axes:

1. **Longest dimension recedes toward the mountains** -- the -Z axis
   (into the distance, away from the camera).
2. **2nd-longest dimension rises toward the sky** -- the +Y axis (up).
3. **3rd-longest dimension spreads left/right** -- the X axis
   (horizontal, across the camera).

Mnemonic: *longest away, next up, next across.* The biggest extent
goes into depth so wide tensors do not sprawl across the viewport;
the camera looks down the -Z corridor.

## Multi-head / "stacked maps" case

For a stack of N maps (e.g. attention `[H, Q, K]`, conv `[C, H, W]`),
the STACK axis (the count of maps, e.g. `H` or `C`) is laid out as a
**row of maps receding toward the mountains** (-Z) -- one map behind
the next -- regardless of whether the stack count is the numerically
longest axis. Within each map the per-map rule applies:

- attention `[H, Q, K]`: heads recede (-Z); within a head the QUERY
  axis is up (+Y, row 0 on top) and the KEY axis is left/right (X).
  Each head is a vertical heatmap wall facing the camera.

This is what "the four heat maps arranged in a row towards the
mountains" means for the Pets per-head attention overlay.

## Status

- [x] Multi-head attention `[H, Q, K]` (`multiHeadStrip` in
  `stage3d.js`): heads recede along -Z; query up, key left/right.
- [x] Rank-2 matrices (`shapeMesh` rank-2 branches, with + without
  values): the LONGER of (rows, cols) recedes toward the mountains
  (-Z), the shorter is width (X); the sheet lies flat so the grid
  reads from above. (A single matrix is not a stack, so it stays
  flat rather than standing up -- the +Y axis is reserved for the
  middle dimension of rank-3 tensors / the per-map axis of stacks.)
- [ ] Conv channel stacks `[C, H, W]` (`convChannelStack`): apply
  the stacked-maps rule (channels recede -Z; H up, W left/right).
- [ ] General rank-1/rank-3+ tensors: audit against the rule.

Each renderer must add a one-line comment citing this doc so the
mapping is not silently re-broken.
