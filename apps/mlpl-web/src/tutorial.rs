use yew::prelude::*;

pub fn toggle_tutorial(lesson: UseStateHandle<Option<usize>>) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| {
        if lesson.is_some() {
            lesson.set(None);
        } else {
            lesson.set(Some(0));
        }
    })
}

pub fn step_lesson(
    lesson: UseStateHandle<Option<usize>>,
    delta: i32,
) -> Callback<web_sys::MouseEvent> {
    Callback::from(move |_| {
        if let Some(cur) = *lesson {
            let next = i32::try_from(cur).unwrap_or(0) + delta;
            if next >= 0 {
                let next_usize = next as usize;
                if next_usize < LESSONS.len() {
                    lesson.set(Some(next_usize));
                }
            }
        }
    })
}

pub fn run_example(
    on_submit: Callback<String>,
    input_value: UseStateHandle<String>,
) -> Callback<String> {
    Callback::from(move |line: String| {
        input_value.set(line.clone());
        on_submit.emit(line);
    })
}

pub struct Lesson {
    pub title: &'static str,
    pub intro: &'static str,
    pub examples: &'static [&'static str],
    pub try_it: &'static str,
}

pub const LESSONS: &[Lesson] = &[
    Lesson {
        title: "Hello Numbers",
        intro: "MLPL starts with simple arithmetic. Numbers are scalars. The four basic operators (+, -, *, /) work as you expect, and parentheses control order of operations.",
        examples: &[
            "1 + 2",
            "3 * 4",
            "10 / 3",
            "2 + 3 * 4",
            "(2 + 3) * 4",
            "-5",
            "-3 + 7",
        ],
        try_it: "Try computing the area of a circle with radius 5: 3.14 * 5 * 5",
    },
    Lesson {
        title: "Arrays",
        intro: "Arrays are MLPL's main data type. Square brackets create a vector. Operators apply element-wise. A scalar is automatically broadcast to match an array's shape.",
        examples: &[
            "[1, 2, 3]",
            "[1, 2, 3] + [4, 5, 6]",
            "[10, 20, 30] - [1, 2, 3]",
            "[1, 2, 3] * 10",
            "100 + [1, 2, 3]",
            "-[1, 2, 3]",
        ],
        try_it: "Create a vector and double every element. Then add 1 to each.",
    },
    Lesson {
        title: "Variables",
        intro: "Bind values to names with =. Variables persist across REPL lines, so you can build up larger computations. Reassignment is allowed.",
        examples: &[
            "x = 42",
            "x + 1",
            "x * x",
            "name = [10, 20, 30]",
            "doubled = name * 2",
            "doubled",
            "x = 100",
            "x",
        ],
        try_it: "Store your favorite number in `n`, then compute n * n - n.",
    },
    Lesson {
        title: "Built-in Functions",
        intro: "Functions are called with parentheses. iota(n) generates the sequence 0..n-1. shape() and rank() inspect arrays. reduce_add and reduce_mul sum or multiply all elements.",
        examples: &[
            "iota(5)",
            "iota(10)",
            "v = iota(6)",
            "shape(v)",
            "rank(v)",
            "reduce_add(v)",
            "reduce_mul([1, 2, 3, 4])",
        ],
        try_it: "Use iota and reduce_add to compute the sum 0 + 1 + ... + 99.",
    },
    Lesson {
        title: "Matrices",
        intro: "reshape turns a flat vector into a multi-dimensional array. transpose flips rows and columns. Reductions can target a specific axis: 0 collapses rows, 1 collapses columns.",
        examples: &[
            "x = iota(12)",
            "m = reshape(x, [3, 4])",
            "m",
            "transpose(m)",
            "shape(m)",
            "reduce_add(m, 0)",
            "reduce_add(m, 1)",
        ],
        try_it: "Build a 2x5 matrix from iota(10) and sum each column.",
    },
    Lesson {
        title: "Linear Algebra",
        intro: "dot(a, b) computes the dot product of two vectors. matmul(A, B) is matrix multiplication: an [m, k] matrix times a [k, n] matrix yields an [m, n] result. These are the building blocks of ML.",
        examples: &[
            "dot([1, 2, 3], [4, 5, 6])",
            "A = reshape(iota(4), [2, 2])",
            "B = reshape([1, 0, 0, 1], [2, 2])",
            "matmul(A, B)",
            "W = reshape([1, 2, 3, 4, 5, 6], [3, 2])",
            "x = reshape([1, 1], [2, 1])",
            "matmul(W, x)",
        ],
        try_it: "Multiply a 2x3 matrix by a 3x1 column vector. What's the result shape?",
    },
    Lesson {
        title: "Math and Activations",
        intro: "exp, log, sqrt, abs apply element-wise. pow does element-wise exponentiation. sigmoid and tanh_fn are smooth activation functions used in neural networks.",
        examples: &[
            "exp(0)",
            "exp([0, 1, 2])",
            "log(exp(1))",
            "sqrt([1, 4, 9, 16])",
            "abs([-3, 0, 5])",
            "pow([2, 3, 4], 2)",
            "sigmoid(0)",
            "sigmoid([-2, -1, 0, 1, 2])",
            "tanh_fn([-1, 0, 1])",
        ],
        try_it: "Apply sigmoid to a vector of negative numbers. What's the range?",
    },
    Lesson {
        title: "Comparisons and Logic",
        intro: "gt, lt, eq compare element-wise and return 0 or 1. Multiplying by a comparison result acts as a mask. mean gives the arithmetic mean of all elements.",
        examples: &[
            "gt([3, 1, 4], 2)",
            "lt([1, 5, 3], 4)",
            "eq([1, 2, 3], [1, 0, 3])",
            "scores = [55, 72, 88, 91, 43]",
            "passed = gt(scores, 60)",
            "passed",
            "mean(passed)",
            "mean(scores)",
        ],
        try_it: "Compute the fraction of elements in iota(10) that are greater than 5.",
    },
    Lesson {
        title: "Loops and Iteration",
        intro: "repeat N { body } runs the body N times. Combined with assignment, this lets you accumulate values or run iterative algorithms. This is the foundation for gradient descent.",
        examples: &[
            "x = 0",
            "repeat 10 { x = x + 1 }",
            "x",
            "total = 0",
            "repeat 100 { total = total + 1 }",
            "total",
            "v = ones([3])",
            "repeat 5 { v = v * 2 }",
            "v",
        ],
        try_it: "Use repeat to compute 2^10 by doubling a variable starting at 1.",
    },
    Lesson {
        title: "Machine Learning: Logistic Regression",
        intro: "Now the payoff. Train a 2-input logistic regression model to learn the AND gate. The model is just matmul + sigmoid. Gradient descent updates the weights to minimize prediction error. After 300 iterations the model is 100% accurate.",
        examples: &[
            "X = [[0,0],[0,1],[1,0],[1,1]]",
            "y = [0, 0, 0, 1]",
            "w = zeros([2])",
            "b = 0",
            "lr = 1.0",
            "n = 4",
            "repeat 300 { z = matmul(X, reshape(w, [2, 1])) + b; pred = sigmoid(z); dz = pred - reshape(y, [4, 1]); dw = reshape(matmul(transpose(X), dz), [2]) / n; db = mean(dz); w = w - lr * dw; b = b - lr * db }",
            "w",
            "b",
            "pred = sigmoid(matmul(X, reshape(w, [2, 1])) + b)",
            "rounded = gt(pred, 0.5)",
            "accuracy = mean(eq(reshape(rounded, [4]), y))",
            "accuracy",
            "svg(reshape(pred, [4]), \"bar\")",
        ],
        try_it: "Change y to [0, 1, 1, 1] and retrain. You just learned the OR gate!",
    },
    Lesson {
        title: "Visualizing Data",
        intro: "svg(data, type) renders an array as an inline SVG diagram. Four diagram types are built in: \"scatter\" for an Nx2 matrix of (x,y) points, \"line\" for a vector or Nx2 matrix, \"bar\" for a vector, and \"heatmap\" for an MxN matrix. The browser REPL displays the SVG inline.",
        examples: &[
            "svg([[0,0],[1,1],[2,4],[3,9],[4,16]], \"scatter\")",
            "svg([1, 3, 2, 5, 4, 6], \"line\")",
            "svg([3, 1, 4, 1, 5, 9, 2, 6], \"bar\")",
            "svg(reshape(iota(25), [5, 5]), \"heatmap\")",
            "ws = iota(20) / 4 - 2",
            "losses = ws * ws + 1",
            "svg(losses, \"line\")",
        ],
        try_it: "Build a vector of squares with iota and pow, then render it as a bar chart.",
    },
];
