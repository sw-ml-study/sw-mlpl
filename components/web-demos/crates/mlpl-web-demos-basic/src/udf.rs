use mlpl_web_demos_types::Demo;

pub const USER_FUNCTIONS: Demo = Demo {
    category: "Basics",
    name: "User-Defined Functions",
    intro: "Define your own functions with def. Names use a colon namespace: \
            u: for user functions, vendor: for packages. Builtins (sin, range, \
            adam) have no colon and cannot be shadowed. Function bodies are full \
            programs, so the building blocks of ML fit naturally: a conditional \
            is an activation, a for loop is a loss over a batch, a while loop is \
            a training run, and recursion is a schedule. Bodies also get early \
            return and access to outer variables.",
    takeaway: "def u:name(args) { body } defines a function; the last expression \
               is the return value and return exits early. Statements are \
               separated by ; or newlines, so a body can branch (if/else), loop \
               (while, for ... in), recurse, and process array arguments. The \
               examples below are the same primitives you would write by hand \
               before reaching for the matmul / adam builtins: relu, an MSE \
               loss, a gradient-descent step, and a decay schedule. Type :fns \
               to list your defined functions.",
    lines: &[
        "# A user function: relu, the classic activation -- a conditional",
        "def u:relu1(x) { if gt(x, 0) { x } else { 0 } }",
        "u:relu1(3)",
        "u:relu1(-2)",
        "# Mean squared error over a vector of residuals -- a for loop over an array arg",
        "def u:mse(errs) { s = 0; n = 0; for r in errs { s = s + r * r; n = n + 1 }; s / n }",
        "u:mse([1, -2, 0.5, -0.5])",
        "# One gradient-descent run minimizing (w - 3)^2 -- a while loop",
        "def u:fit(w0, lr, steps) { w = w0; i = 0; while gt(steps, i) { g = 2 * (w - 3); w = w - lr * g; i = i + 1 }; w }",
        "u:fit(0, 0.1, 50)",
        "# Learning-rate decay schedule lr * rate^steps -- recursion with a base case",
        "def u:decay(lr, rate, steps) { if gt(steps, 0) { u:decay(lr * rate, rate, steps - 1) } else { lr } }",
        "u:decay(0.1, 0.9, 10)",
        "# Numerically safe log(p) for cross-entropy -- early return guards log(0)",
        "def u:safe_log(p) { if gt(p, 0) { return log(p) } else { return -20 } }",
        "u:safe_log(0.5)",
        "u:safe_log(0)",
        "# List all defined functions",
        ":fns",
    ],
};
