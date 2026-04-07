pub struct Demo {
    pub name: &'static str,
    pub lines: &'static [&'static str],
}

pub const DEMOS: &[Demo] = &[
    Demo {
        name: "Basics",
        lines: &[
            "1 + 2",
            "3 * 4",
            "10 / 3",
            "[1, 2, 3] + [4, 5, 6]",
            "[1, 2, 3] * 10",
            "x = [10, 20, 30]",
            "y = x + 1",
            "y",
            "-[1, 2, 3]",
        ],
    },
    Demo {
        name: "Matrix Ops",
        lines: &[
            "x = iota(12)",
            "m = reshape(x, [3, 4])",
            "m",
            "transpose(m)",
            "shape(m)",
            "rank(m)",
            "reduce_add(m, 0)",
            "reduce_add(m, 1)",
        ],
    },
    Demo {
        name: "Math Functions",
        lines: &[
            "exp(0)",
            "exp([0, 1, 2])",
            "log(exp(1))",
            "sqrt([4, 9, 16, 25])",
            "abs([-3, 0, 5])",
            "pow([2, 3, 4], 2)",
            "sigmoid(0)",
            "sigmoid([-2, -1, 0, 1, 2])",
            "tanh_fn([-1, 0, 1])",
        ],
    },
    Demo {
        name: "Logistic Regression",
        lines: &[
            "X = [[0,0],[0,1],[1,0],[1,1]]",
            "y = [0, 0, 0, 1]",
            "w = zeros([2])",
            "b = 0",
            "lr = 1.0",
            "n = 4",
            "repeat 300 { z = matmul(X, reshape(w, [2, 1])) + b; pred = sigmoid(z); dz = pred - reshape(y, [4, 1]); dw = reshape(matmul(transpose(X), dz), [2]) / n; db = mean(dz); w = w - lr * dw; b = b - lr * db }",
            "pred = sigmoid(matmul(X, reshape(w, [2, 1])) + b)",
            "rounded = gt(pred, 0.5)",
            "accuracy = mean(eq(reshape(rounded, [4]), y))",
            "accuracy",
        ],
    },
];
