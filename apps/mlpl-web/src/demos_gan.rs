use crate::demos::Demo;

pub const GAN_2D: Demo = Demo {
    category: "GAN",
    name: "GAN (2D circle)",
    intro: "A Generative Adversarial Network learning to produce points on a \
            unit circle. The Generator maps random 2D noise to 2D points; the \
            Discriminator scores each point as real or fake. They train in \
            alternation: the Discriminator improves at telling real from fake, \
            then the Generator improves at fooling the Discriminator.",
    takeaway: "After 50 steps the Generator produces points near the unit circle \
               even though it never saw the circle directly -- it only received \
               gradient signal from the Discriminator. This adversarial feedback \
               loop is the core GAN idea: two networks competing drives both to \
               improve. The final 'generated' output shows the Generator's best \
               attempt at circle-like 2D points.",
    lines: &[
        "# Real data: 16 points on the unit circle",
        "angles = reshape(iota(16), [16, 1]) * 0.3927",
        "real = concat(cos(angles), sin(angles), 1)",
        "# Generator: noise [16,2] -> hidden [16,8] -> points [16,2]",
        "G = chain(linear(2, 8, 10), relu_layer(), linear(8, 2, 11))",
        "# Discriminator: points [16,2] -> hidden [16,8] -> score [16,1]",
        "D = chain(linear(2, 8, 20), relu_layer(), linear(8, 1, 21))",
        "# Train: 50 steps of alternating D and G updates",
        "train 50 { z = randn(step * 7 + 100, [16, 2]); d_real = sigmoid(apply(D, real)); d_fake = sigmoid(apply(D, apply(G, z))); d_loss = 0 - mean(log(d_real + 0.0001) + log(1 - d_fake + 0.0001)); adam(d_loss, D, 0.002, 0.9, 0.999, 0.00000001); g_loss = 0 - mean(log(sigmoid(apply(D, apply(G, randn(step * 7 + 200, [16, 2])))) + 0.0001)); adam(g_loss, G, 0.002, 0.9, 0.999, 0.00000001); g_loss }",
        "# Show the Generator's output on fresh noise",
        "generated = apply(G, randn(999, [16, 2]))",
        "generated",
    ],
};
