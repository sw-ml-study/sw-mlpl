Step 001: conv2d builtin.

Implement conv2d(input, filters, stride, padding) in a new crates/mlpl-runtime/src/conv_builtins.rs. Input [B, C_in, H, W], filters [C_out, C_in, kH, kW], stride scalar, padding scalar (0=valid, same integer for symmetric padding). Returns [B, C_out, H', W']. Pure nested-loop implementation. Register in builtins.rs dispatch.

TDD: unit test with a known 1-channel 4x4 input, 1-filter 3x3 kernel, stride 1, padding 0 -> expected 2x2 output.