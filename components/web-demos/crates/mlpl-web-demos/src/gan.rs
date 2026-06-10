use mlpl_web_demos_types::Demo;

pub const GAN_2D: Demo = Demo {
    category: "Generative Models",
    name: "GAN (2D circle)",
    intro: "A Generative Adversarial Network learning to produce points on a \
            unit circle. WHAT YOU SEE: two scatter plots -- the FIRST is the \
            target (points on the unit circle), the LAST is what the Generator \
            produced from random noise; it should land on that same circle. The \
            Generator maps 2D noise to 2D points; the Discriminator scores each \
            point real or fake. They train in alternation -- D learns to tell \
            real from fake, then G learns to fool D -- so G is pulled onto the \
            circle WITHOUT ever seeing it directly (its only signal is D's \
            gradient). Each loss is written inline in its adam() call so adam \
            can differentiate it.",
    takeaway: "Compare the two scatters: the target circle (top) vs the \
               Generator's points (bottom), learned purely from the \
               Discriminator's feedback -- it never saw the circle. Two \
               networks competing is the core GAN idea. EXPECT IMPERFECTION: \
               a tiny vanilla GAN like this often suffers MODE COLLAPSE -- the \
               generator covers only part of the circle (an arc or clump) \
               rather than the whole ring. That is a real, well-known GAN \
               failure mode, not a bug in the demo; stabilizing it needs \
               bigger nets / tricks (a GPU-scale follow-up). Diffusion (the \
               'Diffusion (2D points)' demo) is steadier -- it denoises \
               instead of competing.",
    lines: &[
        "n = 64",
        "angles = reshape(range(n), [n, 1]) * 0.0981748          # 64 angles around the circle",
        "real = concat(cos(angles), sin(angles), 1)              # [64,2] points ON the unit circle",
        "svg(real, \"scatter\")                                   # TARGET: the unit circle",
        "G = chain(linear(2, 16, 10), relu_layer(), linear(16, 2, 11))   # Generator: 2D noise -> 2D point",
        "D = chain(linear(2, 16, 20), relu_layer(), linear(16, 1, 21))   # Discriminator: point -> real/fake score",
        "train 1200 { z = randn(step * 7 + 100, [n, 2]) ; adam(0 - mean(log(sigmoid(apply(D, real)) + 0.0001) + log(1 - sigmoid(apply(D, apply(G, z))) + 0.0001)), D, 0.002, 0.9, 0.999, 0.00000001) ; adam(0 - mean(log(sigmoid(apply(D, apply(G, z))) + 0.0001)), G, 0.002, 0.9, 0.999, 0.00000001) ; adam(0 - mean(log(sigmoid(apply(D, apply(G, z))) + 0.0001)), G, 0.002, 0.9, 0.999, 0.00000001) ; 0 }  # alternate: train D once, then G twice -- losses INLINE so adam differentiates them",
        "gen = apply(G, randn(999, [n, 2]))                       # generate from fresh noise",
        "svg(gen, \"scatter\")                                    # GENERATED: should land on the circle above",
    ],
};
