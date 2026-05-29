use mlpl_web_demos_types::Demo;

pub const USER_FUNCTIONS: Demo = Demo {
    category: "Basics",
    name: "User-Defined Functions",
    intro: "Define your own functions with def. Names use a colon namespace: \
            u: for user functions, vendor: for packages. Builtins (sin, range, \
            adam) have no colon and cannot be shadowed. Functions support \
            recursion, early return, and access to outer variables.",
    takeaway: "def u:name(args) { body } defines a function. The last \
               expression is the return value; use return for early exit. \
               Parameters shadow outer variables during the call and restore \
               after. Type :fns to list your defined functions.",
    lines: &[
        "# Define a simple function",
        "def u:double(x) { x * 2 }",
        "u:double(21)",
        "# Multiple parameters",
        "def u:add(a, b) { a + b }",
        "u:add(10, 32)",
        "# Use builtins inside UDFs",
        "def u:circle_area(r) { pi() * r * r }",
        "u:circle_area(5)",
        "# Recursion: factorial",
        "def u:fact(n) { if gt(n, 1) { n * u:fact(n - 1) } else { 1 } }",
        "u:fact(6)",
        "# Early return",
        "def u:safe_div(a, b) { if eq(b, 0) { return 0 } else { a / b } }",
        "u:safe_div(10, 3)",
        "u:safe_div(10, 0)",
        "# List all defined functions",
        ":fns",
    ],
};
