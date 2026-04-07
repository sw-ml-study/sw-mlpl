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
    Lesson {
        title: "Visualizing Analyses",
        intro: "High-level helpers compute the right view of your data and render a complete diagram in one call. hist makes a histogram, scatter_labeled colors points by cluster id, loss_curve plots a training loss vector, confusion_matrix renders a KxK heatmap of class predictions vs actuals, and boundary_2d renders a 2D classifier decision surface with training points overlaid.",
        examples: &[
            "hist([1, 2, 2, 3, 3, 3, 4, 4, 5], 5)",
            "scatter_labeled([[0,0],[1,1],[0,1],[1,0]], [0, 0, 1, 1])",
            "loss_curve([5, 3, 2, 1.5, 1.0, 0.7, 0.5, 0.4, 0.3, 0.25])",
            "confusion_matrix([0, 1, 2, 1, 0], [0, 1, 1, 1, 0])",
        ],
        try_it: "Generate a histogram of iota(20) with 5 bins.",
    },
    Lesson {
        title: "Unsupervised: K-Means",
        intro: "K-Means clustering finds K groups in unlabeled data by repeating two steps: (1) assign each point to its nearest cluster center, and (2) move each center to the mean of its assigned points. We use blobs() to make a synthetic dataset, then express the whole Lloyd iteration with matmul, reduce_add, and argmax -- no per-point loops. The assignment mask is built from eq(iota(K), labels) broadcast via matmul.",
        examples: &[
            "D = blobs(7, 20, [[0, 0], [4, 4], [-4, 4]])",
            "X = matmul(D, [[1,0],[0,1],[0,0]])",
            "C = [[1, 1], [3, 3], [-3, 3]]",
            "sqX = reshape(reduce_add(X*X, 1), [60, 1])",
            "sqC = reshape(reduce_add(C*C, 1), [1, 3])",
            "dists = matmul(sqX, ones([1, 3])) + matmul(ones([60, 1]), sqC) - 2 * matmul(X, transpose(C))",
            "clus = argmax(-1 * dists, 1)",
            "scatter_labeled(X, clus)",
        ],
        try_it: "Run a full K-Means from the K-Means demo and compare the learned centers to the true centers [[0,0],[4,4],[-4,4]].",
    },
    Lesson {
        title: "Dimensionality Reduction: PCA",
        intro: "Principal Component Analysis finds the direction along which a dataset varies the most. First center the data by subtracting the column means. The covariance matrix is Xc^T Xc / n. Its dominant eigenvector is the first principal component. Power iteration -- repeatedly multiplying a unit vector by the covariance and renormalizing -- converges to that eigenvector without needing an eigensolver. Projecting the centered data onto the principal axis gives a 1D coordinate you can color points by.",
        examples: &[
            "Xraw = randn(1, [60, 2])",
            "X = matmul(Xraw, [[1, 2], [0, 0.3]])",
            "cm = reduce_add(X, 0) / 60",
            "Xc = X - matmul(ones([60, 1]), reshape(cm, [1, 2]))",
            "Cov = matmul(transpose(Xc), Xc) / 60",
            "v = [1, 0]",
            "repeat 10 { v = matmul(Cov, v); v = v / sqrt(dot(v, v)) }",
            "v",
        ],
        try_it: "Change the mixing matrix to [[1, 0], [0, 3]] and verify that the principal axis comes out parallel to (0, 1).",
    },
    Lesson {
        title: "Multi-class Classification",
        intro: "Softmax regression generalizes logistic regression to K classes. Logits Z = X W + b are turned into row-normalized probabilities P by softmax(Z, 1), which subtracts the row max for stability and exponentiates. With one-hot labels Y, the cross-entropy gradient is elegant: dZ = P - Y, so dW = X^T (P - Y) / N and db = mean(P - Y, 0). At prediction time, argmax(P, 1) picks the most probable class.",
        examples: &[
            "Z = [[1, 2, 3], [2, 0, -1]]",
            "softmax(Z, 1)",
            "one_hot([0, 2, 1, 0], 3)",
        ],
        try_it: "Run the Softmax Classifier demo and read off the diagonal of the confusion matrix -- those are the correctly classified points.",
    },
    Lesson {
        title: "Going Non-Linear: A Tiny MLP",
        intro: "A single-layer softmax classifier can only draw straight lines. Adding a hidden layer with a non-linear activation lets the network bend the decision boundary around the data. A 2 -> 8 -> 2 MLP with tanh activations learns the XOR pattern that the linear model from the previous lesson cannot. The backward pass uses the chain rule: dZ2 = P - Y at the output, and the hidden-layer gradient is dZ1 = (dZ2 W2^T) * (1 - H*H), which is the tanh derivative applied element-wise.",
        examples: &[
            "D = blobs(3, 20, [[-2,-2],[2,2],[-2,2],[2,-2]])",
            "X = matmul(D, [[1,0],[0,1],[0,0]])",
            "raw = reshape(matmul(D, [[0],[0],[1]]), [80])",
            "y = gt(raw, 1.5)",
            "Y = one_hot(y, 2)",
            "W1 = randn(5, [2, 8]) * 0.5",
            "W2 = randn(6, [8, 2]) * 0.5",
        ],
        try_it: "Try running the Tiny MLP demo, then the Softmax Classifier demo on the same data -- compare the boundary shapes.",
    },
    Lesson {
        title: "Attention Patterns",
        intro: "Scaled dot-product attention is a single matrix expression: A = softmax(Q K^T / sqrt(d), 1). Each row of A is a probability distribution over the keys -- a weight map for one query. The scaling by sqrt(d) keeps the scores from growing with the key dimension so softmax doesn't saturate. When K = Q (self-attention), each token's highest score is on itself, so the diagonal of A dominates. Rendering A as a heatmap shows the full attention pattern at a glance.",
        examples: &[
            "Q = randn(17, [6, 4])",
            "K = randn(23, [6, 4])",
            "scores = matmul(Q, transpose(K)) / sqrt(4)",
            "A = softmax(scores, 1)",
            "svg(A, \"heatmap\")",
        ],
        try_it: "Set K = Q and re-render the heatmap -- notice how the diagonal becomes the brightest cell in each row.",
    },
];
