Step 002: pool2d builtin.

Implement pool2d(input, size, mode) in conv_builtins.rs. Input [B, C, H, W], size [pH, pW] as array arg, mode as string 'max' or 'avg'. Returns [B, C, H/pH, W/pW]. Unit tests for max and avg pooling.