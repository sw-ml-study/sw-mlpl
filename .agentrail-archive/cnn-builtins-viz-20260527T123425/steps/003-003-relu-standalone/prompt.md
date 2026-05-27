Step 003: standalone relu(x) builtin.

Add relu(x) = max(0, x) as an elementwise builtin in math_builtins.rs. Separate from the model DSL relu_layer(). Update lang-reference.md and help text.