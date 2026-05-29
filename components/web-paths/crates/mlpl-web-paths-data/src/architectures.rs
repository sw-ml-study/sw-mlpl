use crate::types::{LearningPath, Step};

pub(super) const PATH_ARCHITECTURE_ZOO__FROM_PIXELS_TO_LANGUAGE: LearningPath = LearningPath {
    title: "Architecture Zoo: from pixels to language",
    blurb: "How machines learn to see, generate, and remember. Walk the four major ML architecture families in dependency order: CNN (spatial structure), RNN/LSTM (sequential memory), Autoencoder/GAN (generative models), and Transformer (global attention). Each group pairs a glossary orientation with runnable demos.",
    steps: &[
        Step::Note {
            title: "What is an architecture?",
            body: "Each ML architecture exploits a different kind of structure in data. A CNN exploits spatial locality (nearby pixels matter most). An RNN exploits sequential order (what came before matters). An autoencoder exploits compressibility (data lives on a low-dimensional manifold). A GAN exploits adversarial feedback. A transformer exploits long-range attention. This path walks them in order of increasing complexity.",
        },
        // --- Group 1: See (spatial structure) ---
        Step::Note {
            title: "Group 1: See (spatial structure)",
            body: "CNNs exploit the fact that nearby pixels are more related than distant ones. A small sliding filter (kernel) scans the image, producing a feature map that detects edges, textures, or shapes. Pooling then shrinks the spatial dimensions, keeping the strongest signals. Stacking conv + pool layers builds a hierarchy: edges -> textures -> parts -> objects.",
        },
        Step::Glossary {
            term: "Convolution",
            why: "The core CNN operation: a small filter slides across the input, computing dot products at each position. MLPL: conv2d(input, filters, stride, padding).",
        },
        Step::Glossary {
            term: "Feature map",
            why: "The output of a convolutional layer: one 2D grid per filter, each highlighting a different pattern in the input.",
        },
        Step::Glossary {
            term: "Pooling",
            why: "Controlled information loss: shrink spatial dimensions by taking the max or average over small windows. MLPL: pool2d(input, size, mode).",
        },
        Step::Demo {
            name: "CNN (simple)",
            why: "A minimal conv2d -> relu -> pool2d -> flatten -> linear pipeline on a 4x4 synthetic input. Shows the shape transformations at each stage.",
        },
        Step::Diagram {
            slug: "08_cnn",
            why: "The canonical CNN diagram: input -> conv+relu -> pool -> conv+relu -> pool -> flatten -> FC. MLPL ships conv2d and pool2d builtins (saga 39).",
        },
        // --- Group 2: Remember (sequential structure) ---
        Step::Note {
            title: "Group 2: Remember (sequential order)",
            body: "RNNs process sequences one element at a time, carrying a hidden state forward like a running summary. At each step the cell reads one new input and updates its memory. The same weights are reused at every step (weight sharing). The problem: vanilla RNNs forget early inputs after ~10-20 steps because gradients vanish through repeated tanh squashing.",
        },
        Step::Glossary {
            term: "RNN (Recurrent Neural Network)",
            why: "The basic recurrent cell: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + bias). MLPL: rnn_cell(input, hidden, W_ih, W_hh, bias).",
        },
        Step::Demo {
            name: "RNN (sequence)",
            why: "A 5-step unrolled RNN showing hidden state evolution. Each rnn_cell call is one tick; watch the hidden vector change at each time step.",
        },
        Step::Glossary {
            term: "Vanishing Gradient",
            why: "Why vanilla RNNs forget: gradients shrink exponentially through repeated tanh, losing signal from early inputs.",
        },
        Step::Glossary {
            term: "LSTM (Long Short-Term Memory)",
            why: "Gated RNN with separate cell state: forget/input/output gates control what to keep, add, and expose. Solves vanishing gradients.",
        },
        Step::Demo {
            name: "LSTM (sequence memory)",
            why: "A 10-step unrolled LSTM -- twice the length of the RNN demo. The cell state carries information across many steps without washing out.",
        },
        Step::Glossary {
            term: "Hidden State",
            why: "The fixed-size vector a recurrent network carries between time steps -- the network's memory of everything it has seen so far.",
        },
        Step::Diagram {
            slug: "10_rnn",
            why: "Hidden state passed through time. MLPL ships rnn_cell (saga 41).",
        },
        Step::Diagram {
            slug: "11_lstm",
            why: "The four-gate LSTM cell. MLPL ships lstm_cell (saga 41).",
        },
        // --- Group 3: Create (generative structure) ---
        Step::Note {
            title: "Group 3: Create (generative models)",
            body: "Generative models learn to produce new data that resembles the training set. An autoencoder compresses data through a bottleneck and reconstructs it -- the bottleneck forces the model to learn the essential structure. A GAN pits a Generator against a Discriminator in an adversarial game -- the Generator improves by fooling the Discriminator.",
        },
        Step::Glossary {
            term: "Bottleneck (autoencoder)",
            why: "The narrow layer in an autoencoder that forces compression. The network must learn which features matter enough to preserve.",
        },
        Step::Demo {
            name: "Autoencoder (simple)",
            why: "Encoder shrinks 8 dimensions to 3, decoder expands back to 8. Train with MSE reconstruction loss. The bottleneck vector is the learned compressed representation.",
        },
        Step::Glossary {
            term: "Reconstruction Error",
            why: "How well the autoencoder recovers the original input from the bottleneck. Lower = better compression without information loss.",
        },
        Step::Glossary {
            term: "GAN (Generative Adversarial Network)",
            why: "Two networks in competition: Generator creates fakes, Discriminator catches them. At equilibrium the Generator produces realistic data.",
        },
        Step::Demo {
            name: "GAN (2D circle)",
            why: "Generator learns to produce unit-circle points from random noise. 50 alternating adam steps train Generator and Discriminator in competition.",
        },
        Step::Glossary {
            term: "Latent Space",
            why: "The internal representation space -- hidden-layer activations that are often more semantically structured than the raw input.",
        },
        // --- Group 4: Attend (global structure) ---
        Step::Note {
            title: "Group 4: Attend (global attention)",
            body: "Transformers replaced RNNs for most sequence tasks by using attention instead of recurrence. Attention lets every position look at every other position in parallel -- no sequential bottleneck, no vanishing gradients. The cost is O(n^2) in sequence length, but the parallelism makes it faster on modern hardware.",
        },
        Step::Diagram {
            slug: "12_attention",
            why: "The attention formula: softmax(QK^T / sqrt(d_k)) V. The diagram that makes it click.",
        },
        Step::Lesson {
            title: "Self-Attention from Scratch",
            why: "Build one head of attention from primitives: three projections (Q/K/V), scaled dot-product, softmax, weighted sum.",
        },
        Step::Demo {
            name: "Multi-Head Attention from Scratch",
            why: "Multiple attention heads run in parallel, each on a d_k slab. The from-scratch implementation shows how heads split and recombine.",
        },
        Step::Diagram {
            slug: "17_gpt_decoder_only",
            why: "Stacked causal-self-attention blocks. This IS the Tiny LM architecture.",
        },
        Step::Demo {
            name: "Tiny LM Generate",
            why: "End-to-end: BPE tokenizer + 1-layer transformer trained 30 steps, then sampled to generate text. The smallest program that learns to talk.",
        },
        Step::Note {
            title: "Why transformers replaced RNNs",
            body: "RNNs process tokens one at a time -- each step waits for the previous one. Transformers process all tokens in parallel via attention. RNNs compress history into a fixed-size hidden state that forgets; transformers let every token attend to every other token directly. The tradeoff: transformers need O(n^2) memory for n tokens, but modern hardware makes that worthwhile up to ~128K tokens.",
        },
        Step::Note {
            title: "Where to go from here",
            body: "You have seen the four major architecture families. The 'Zero to LLM' path goes deeper into transformers. The 'Build a transformer from primitives' path constructs encoder and decoder blocks step by step. The 'Vision Transformers' path applies attention to image patches. Each architecture here has a dedicated demo you can re-run and modify in the REPL.",
        },
    ],
};

pub(super) const PATH_OPTIMIZERS___REGULARIZATION: LearningPath = LearningPath {
    title: "Optimizers & Regularization",
    blurb: "How to make training faster, more stable, and less likely to overfit. Walk from hand-rolled gradient descent through momentum, Adam, learning rate schedules, regularization techniques, and parameter-efficient fine-tuning (LoRA). Each step builds on the last.",
    steps: &[
        Step::Note {
            title: "Why optimizers matter",
            body: "Gradient descent is simple: subtract the gradient times a learning rate. But vanilla SGD is slow (zig-zags in narrow valleys), sensitive to learning rate, and has no memory of previous steps. Every optimizer improvement since 1986 has been about fixing one of these problems.",
        },
        Step::Glossary {
            term: "Gradient descent",
            why: "The foundation: w = w - lr * grad(loss, w). Everything else is a refinement of this update rule.",
        },
        Step::Lesson {
            title: "Machine Learning: Logistic Regression",
            why: "Hand-rolled gradient descent on 4 data points. See the raw update rule before any optimizer hides it.",
        },
        Step::Demo {
            name: "Loss Curve",
            why: "Sweep a weight across 25 values and plot MSE. The parabolic bowl is what gradient descent walks down.",
        },
        Step::Glossary {
            term: "Momentum SGD",
            why: "Add a velocity term: v = beta*v + grad; w = w - lr*v. Smooths zig-zags, accelerates in consistent gradient directions. MLPL: momentum_sgd(loss, params, lr, beta).",
        },
        Step::Glossary {
            term: "Adam",
            why: "Per-parameter adaptive learning rates: first moment (momentum) + second moment (RMSProp) + bias correction. The default optimizer since ~2016. MLPL: adam(loss, params, lr, b1, b2, eps).",
        },
        Step::Lesson {
            title: "Optimizers and Schedules",
            why: "Compare momentum_sgd and adam side-by-side. See how cosine_schedule and linear_warmup shape the learning rate over training.",
        },
        Step::Glossary {
            term: "Learning Rate Schedules",
            why: "Start high (fast early progress), decay low (fine-tune near the minimum). Cosine annealing and linear warmup are the two MLPL ships.",
        },
        Step::Glossary {
            term: "Learning rate",
            why: "The single most important hyperparameter. Too high: diverge. Too low: stuck. Schedules automate the tradeoff.",
        },
        Step::Note {
            title: "Regularization: fighting overfitting",
            body: "A model that memorizes training data but fails on new data is overfitting. Regularization techniques add friction that prevents memorization: dropout randomly silences neurons, weight decay shrinks weights toward zero, early stopping halts training before the model starts memorizing. The goal is a model that generalizes.",
        },
        Step::Glossary {
            term: "Overfitting / Underfitting",
            why: "Training loss keeps falling but validation loss rises. The model has memorized the training data.",
        },
        Step::Glossary {
            term: "Regularization",
            why: "Any technique that reduces overfitting: dropout, weight decay, data augmentation, early stopping.",
        },
        Step::Glossary {
            term: "Dropout",
            why: "Randomly zero activations during training. Forces redundant representations. MLPL does not ship dropout (models are small enough).",
        },
        Step::Glossary {
            term: "Batch Normalization",
            why: "Normalize activations per mini-batch. Stabilizes training, enables higher learning rates. MLPL ships rms_norm (the transformer variant).",
        },
        Step::Note {
            title: "Parameter-efficient fine-tuning",
            body: "Fine-tuning a large pre-trained model means updating all its parameters on a new task. LoRA freezes the base model and trains only small rank-r adapter matrices at each layer -- 10,000x fewer trainable parameters, similar quality. This makes fine-tuning practical on consumer hardware.",
        },
        Step::Glossary {
            term: "LoRA (Low-Rank Adaptation)",
            why: "Freeze base weights, train rank-r adapters. MLPL: lora(model, rank, alpha, seed).",
        },
        Step::Lesson {
            title: "LoRA Fine-Tuning",
            why: "The MLPL lesson: freeze a pre-trained model, add LoRA adapters, compare loss curves with and without freezing.",
        },
        Step::Note {
            title: "The optimizer landscape",
            body: "SGD -> Momentum -> Adam is the historical arc. Schedules (cosine, warmup) shape the learning rate over time. Regularization (dropout, weight decay, batch norm) prevents overfitting. LoRA makes fine-tuning cheap. Modern training combines all of these: Adam with cosine schedule, warmup, and LoRA adapters on a frozen base model.",
        },
    ],
};

pub(super) const PATH_TRAINING_PARADIGMS: LearningPath = LearningPath {
    title: "Training Paradigms",
    blurb: "How do models learn? Four paradigms, each exploiting a different kind of signal: labeled data (supervised), structure in unlabeled data (unsupervised), self-generated labels (self-supervised), and reward from an environment (reinforcement). This path walks them in historical order with runnable demos where MLPL has the primitives.",
    steps: &[
        Step::Note {
            title: "What is a training paradigm?",
            body: "A training paradigm is the answer to 'where does the learning signal come from?' Supervised learning has a teacher (labeled data). Unsupervised learning has no teacher (find structure). Self-supervised learning manufactures its own labels from the data. Reinforcement learning has a reward signal from an environment. Each paradigm suits different problems.",
        },
        // --- Group 1: Supervised ---
        Step::Note {
            title: "Group 1: Supervised learning",
            body: "The classic: given input X and label y, minimize the loss between the model's prediction and the true label. Gradient descent on a differentiable loss function. Every classification and regression demo in MLPL uses supervised learning. The paradigm that powered ML from the 1950s through the 2010s.",
        },
        Step::Glossary {
            term: "Supervised learning",
            why: "The paradigm: labeled data + loss function + gradient descent.",
        },
        Step::Lesson {
            title: "Machine Learning: Logistic Regression",
            why: "The hello-world: fit weights to labeled data with hand-rolled gradient descent.",
        },
        Step::Demo {
            name: "Moons MLP",
            why: "A 2-layer MLP trained with adam + cross-entropy on the two-moons dataset. The canonical supervised pipeline.",
        },
        Step::Glossary {
            term: "Cross entropy",
            why: "The standard classification loss: -sum(y * log(pred)). Differentiable end-to-end through grad.",
        },
        Step::Glossary {
            term: "Adam",
            why: "The default optimizer: per-parameter adaptive learning rates with momentum.",
        },
        // --- Group 2: Unsupervised ---
        Step::Note {
            title: "Group 2: Unsupervised learning",
            body: "No labels at all. The model discovers structure in the data: clusters (K-Means), principal axes (PCA), local neighborhoods (t-SNE, UMAP). The loss is internal -- distance to centroids, variance explained, KL divergence. Unsupervised methods are often used as preprocessing (dimensionality reduction before classification) or exploration (what groups exist in my data?).",
        },
        Step::Glossary {
            term: "Unsupervised learning",
            why: "The paradigm: no labels, learn structure from data geometry.",
        },
        Step::Lesson {
            title: "Unsupervised: K-Means",
            why: "Assign points to K clusters, move centers to cluster means, repeat. The simplest unsupervised algorithm.",
        },
        Step::Lesson {
            title: "Dimensionality Reduction: PCA",
            why: "Find the axes of maximum variance. Unsupervised: no labels, just geometry.",
        },
        Step::Demo {
            name: "Dim-reduction zoo",
            why: "Five unsupervised projections side-by-side: PCA, t-SNE, UMAP, MDS, random projection.",
        },
        // --- Group 3: Self-supervised ---
        Step::Note {
            title: "Group 3: Self-supervised learning",
            body: "Manufacture labels from the data itself. Mask a word and predict it (BERT). Predict the next token (GPT). Crop an image patch and predict its embedding (JEPA). The key insight: you can create billions of labeled examples for free from unlabeled data. This paradigm powers all modern foundation models -- they pre-train self-supervised, then fine-tune supervised.",
        },
        Step::Glossary {
            term: "Self-supervised learning",
            why: "The paradigm: labels come from the data itself (masked prediction, next-token, contrastive pairs).",
        },
        Step::Demo {
            name: "Tiny LM Generate",
            why: "Next-token prediction: the model predicts each token given the previous ones. The labels are the text itself, shifted by one position.",
        },
        Step::Lesson {
            title: "Self-Attention from Scratch",
            why: "Attention is the mechanism that makes self-supervised pre-training scale. Every position attends to every other position in parallel.",
        },
        Step::Glossary {
            term: "BERT",
            why: "Masked language modeling: mask 15% of tokens, predict them from context. The encoder-side self-supervised paradigm.",
        },
        // --- Group 4: Reinforcement ---
        Step::Note {
            title: "Group 4: Reinforcement learning",
            body: "An agent takes actions in an environment and receives rewards. The learning signal is sparse and delayed -- the agent must explore to discover which actions lead to high reward. No labeled examples at all; the agent generates its own training data by interacting with the environment. RLHF applies this to LLMs: the 'environment' is human preference judgments.",
        },
        Step::Glossary {
            term: "RLHF (Reinforcement Learning from Human Feedback)",
            why: "The paradigm applied to LLM alignment: SFT -> reward model -> PPO. Human preferences replace environment rewards.",
        },
        Step::Glossary {
            term: "Reward Hacking",
            why: "The dark side of RL: the agent optimizes the reward signal rather than the intended behavior. Goodhart's Law in action.",
        },
        Step::Note {
            title: "The paradigm spectrum",
            body: "These paradigms are not mutually exclusive. Modern training often combines them: pre-train self-supervised (GPT-style next-token), fine-tune supervised (SFT on instruction data), then align with RL (RLHF/DPO on human preferences). The paradigm tells you where the gradient comes from at each stage.",
        },
    ],
};
