pub fn help_text() -> String {
    [
        "MLPL v0.4 -- Array Programming Language for ML",
        "",
        "Syntax:",
        "  42              scalar literal",
        "  [1, 2, 3]       array literal",
        "  x = expr        assignment",
        "  a + b           arithmetic (+, -, *, /)",
        "  func(args)      function call",
        "  repeat N { }    loop N times",
        "",
        "Built-in functions:",
        "  iota, shape, rank, reshape, transpose",
        "  reduce_add, reduce_mul (with optional axis)",
        "  dot, matmul",
        "  exp, log, sqrt, abs, pow",
        "  sigmoid, tanh_fn",
        "  gt, lt, eq, mean",
        "  zeros, ones, fill",
        "",
        "Commands:",
        "  :help   show this help",
        "  :clear  reset all variables",
    ]
    .join("\n")
}
