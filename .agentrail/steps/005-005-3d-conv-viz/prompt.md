Step 005: 3D conv visualization.

Update stage3d.js to render rank-4 tensors [B, C, H, W] as stacked heatmap layers (one per channel). Each layer is a colored grid showing element values. Conv filters rendered as small grids. Pages rebuild required.