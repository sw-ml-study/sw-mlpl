use crate::demos::Demo;

pub const AUTOENCODER: Demo = Demo {
    category: "Autoencoder",
    name: "Autoencoder (simple)",
    intro: "An autoencoder compresses input through a narrow bottleneck, then \
            reconstructs it. The encoder shrinks 8 dimensions to 3; the decoder \
            expands back to 8. Training minimizes reconstruction error (MSE). \
            Watch the loss drop as the network learns to compress and decompress.",
    takeaway: "After training, the 3-element bottleneck captures the essential \
               structure of the 8-element input. The reconstruction isn't perfect \
               but close -- that gap is the information the bottleneck couldn't \
               retain. In :3d mode, the shapes tell the story: wide -> narrow -> wide.",
    lines: &[
        "# Synthetic 8-element input data (one sample)",
        "x = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]",
        "",
        "# Encoder: 8 -> 5 -> 3 (bottleneck)",
        "enc = chain(linear(8, 5, 42), relu_layer(), linear(5, 3, 43))",
        "",
        "# Decoder: 3 -> 5 -> 8 (reconstruct)",
        "dec = chain(linear(3, 5, 44), relu_layer(), linear(5, 8, 45))",
        "",
        "# Full autoencoder: encoder then decoder",
        "ae = chain(enc, dec)",
        "",
        "# Before training: reconstruction is random",
        "recon_before = apply(ae, x)",
        "",
        "# Train: minimize MSE between input and reconstruction",
        "train 30 { adam(reduce_add((apply(ae, x) - x) * (apply(ae, x) - x), 0) / 8, ae, 0.01, 0.9, 0.999, 0.00000001) }",
        "",
        "# After training: reconstruction should be close to input",
        "recon_after = apply(ae, x)",
        "",
        "# The bottleneck: what the encoder compressed x into",
        "latent = apply(enc, x)",
        "latent                                          # just 3 numbers!",
        "",
        "# Compare input vs reconstruction",
        "x",
        "recon_after",
    ],
};
