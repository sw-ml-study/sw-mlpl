use mlpl_web_demos_types::Demo;

pub const SIMPLE_CNN: Demo = Demo {
    category: "Vision",
    name: "CNN (simple)",
    intro: "A convolutional neural network pipeline on a tiny 8x8 synthetic image. \
            Shows how conv2d shrinks the spatial dimensions while relu clips negatives \
            and pool2d downsamples. The shapes are the story: [1,1,8,8] -> conv2d -> \
            [1,2,6,6] -> relu -> pool2d -> [1,2,3,3] -> conv2d -> [1,4,1,1] -> reshape -> [4].",
    takeaway: "Each conv2d reduces spatial size by (kernel-1) and increases channels. \
               pool2d halves the grid. relu zeros the negatives. By the end, the 8x8 \
               image is compressed to a 4-element feature vector -- that's what a \
               classifier reads.",
    lines: &[
        "# Synthetic 8x8 grayscale image (batch=1, channels=1)",
        "img = reshape(range(64) / 64, [1, 1, 8, 8])",
        "shape(img)                                                # [1, 1, 8, 8]",
        "# First conv layer: 2 filters of size 3x3",
        "f1 = reshape(randn(42, [18]), [2, 1, 3, 3])",
        "c1 = conv2d(img, f1, 1, 0)                              # [1, 2, 6, 6]",
        "shape(c1)",
        "a1 = relu(c1)                                            # clip negatives",
        "p1 = pool2d(a1, [2, 2], 1)                               # max pool -> [1, 2, 3, 3]",
        "shape(p1)",
        "# Second conv layer: 4 filters of size 3x3",
        "f2 = reshape(randn(7, [72]), [4, 2, 3, 3])",
        "c2 = conv2d(p1, f2, 1, 0)                               # [1, 4, 1, 1]",
        "shape(c2)",
        "# Flatten to a feature vector",
        "features = reshape(c2, [4])",
        "features                                                  # 4-element feature vector",
    ],
};
