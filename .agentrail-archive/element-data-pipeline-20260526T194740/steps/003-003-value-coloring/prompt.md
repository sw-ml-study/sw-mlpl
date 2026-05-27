Step 003: Value-colored sculptures.

Update stage3d.js to use element values for coloring:
- Scalar sphere: color encodes value (blue negative, white zero, red positive)
- Vector bar: each element is a thin column, height=abs(value), color=blue-white-red diverging
- Matrix: flat grid of colored cells, color=value on diverging colormap

When values are present, the sculpture shows actual data. When absent (values=null), falls back to the current solid-color shape.

For large arrays with only summary data, color the mesh by the mean and show a miniature histogram sprite.

Pages rebuild required.