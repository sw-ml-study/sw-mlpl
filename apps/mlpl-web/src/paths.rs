//! Learning paths: curated ordered walks through the
//! tutorial / demo / diagram / glossary surfaces.
//!
//! A `LearningPath` is just a list of `Step`s, each of which
//! references existing content by name (lessons by title,
//! demos by name, diagrams by filename slug, glossary entries
//! by exact term). The walker view (`paths_view::PathsView`)
//! renders each step with a path-specific "why this is here"
//! framing and -- for lessons / demos -- a button that jumps
//! to the corresponding tab. Diagrams and glossary entries
//! render inline.
//!
//! Paths are pure data: adding a new path is one entry in
//! `PATHS` below, no UI changes needed.

#[derive(Clone, Copy, PartialEq)]
pub struct LearningPath {
    pub title: &'static str,
    pub blurb: &'static str,
    pub steps: &'static [Step],
}

#[derive(Clone, Copy, PartialEq)]
pub enum Step {
    /// A tutorial lesson, looked up by exact title.
    Lesson {
        title: &'static str,
        why: &'static str,
    },
    /// A demo, looked up by exact name.
    Demo {
        name: &'static str,
        why: &'static str,
    },
    /// A diagram, looked up by filename slug (matching the
    /// numbered `<slug>.svg` files in `diagrams/`).
    Diagram {
        slug: &'static str,
        why: &'static str,
    },
    /// A glossary entry, looked up by exact term (matching
    /// `## TermName` headers in `docs/glossary.md`).
    Glossary {
        term: &'static str,
        why: &'static str,
    },
    /// A path-orientation note that does not reference
    /// existing content. Shown as a small framing card.
    Note {
        title: &'static str,
        body: &'static str,
    },
}

pub const PATHS: &[LearningPath] = &[
    LearningPath {
        title: "A chronological history of ML",
        blurb: "Twenty-four eras from the first formal neuron (1943) to state-space models (2023). Each stop names the foundational paper, the year, the key idea, and what MLPL can demonstrate today. Walk the whole spine in ~30 minutes, or jump to any era.",
        steps: &[
            Step::Note {
                title: "How to read this path",
                body: "Each era names the paper, the year of first publication, and when the idea saw wider adoption or key refinements. Where MLPL has the primitives, there is a runnable demo or lesson. Where it does not, there is a glossary entry and a note on what would be needed.",
            },
            // 1. McCulloch-Pitts (1943)
            Step::Note {
                title: "1943 -- McCulloch-Pitts neuron",
                body: "The first formal model of a neuron: a thresholded weighted sum of binary inputs (McCulloch and Pitts, 1943). Not learned -- the weights were set by hand. The idea that computation could be described in terms of networks of simple units launched both AI and computational neuroscience. Wider impact: 1950s-60s, when Rosenblatt added learning.",
            },
            // 2. Hebbian learning (1949)
            Step::Note {
                title: "1949 -- Hebbian learning",
                body: "\"Neurons that fire together, wire together\" (Hebb, 1949). The first learning rule: strengthen connections between co-activated neurons. Unsupervised -- no teacher signal. Foundation for later associative memory models (Hopfield 1982) and modern contrastive learning. MLPL does not ship Hebbian updates; the optimizer set starts at SGD.",
            },
            // 3. Perceptron (1958)
            Step::Note {
                title: "1958 -- The Perceptron",
                body: "The first machine that learned from data (Rosenblatt, 1958). One layer of weights + a hard threshold, trained by the perceptron convergence theorem: if the data is linearly separable, the algorithm converges in finite steps. Minsky and Papert (1969) proved it could not solve XOR, triggering the first AI winter. Wider adoption: immediate (military funding), then collapse (1969-1986).",
            },
            Step::Glossary {
                term: "Perceptron",
                why: "The single-layer ancestor of every neural network. MLPL's logistic regression demo is a continuous relaxation of the same idea.",
            },
            // 4. Backpropagation (1986)
            Step::Note {
                title: "1986 -- Backpropagation",
                body: "The chain rule applied systematically to multi-layer networks (Rumelhart, Hinton, Williams, 1986). Earlier derivations existed (Werbos 1974, Linnainmaa 1970), but the 1986 Nature paper demonstrated it on practical problems and ended the first AI winter. Every modern neural network trains with backprop. Refinements: automatic differentiation frameworks (Theano 2010, TensorFlow 2015, PyTorch 2016).",
            },
            Step::Lesson {
                title: "Why backprop?",
                why: "The lesson that explains why reverse-mode autodiff is the right choice for neural nets: one backward pass computes all gradients, regardless of parameter count.",
            },
            Step::Lesson {
                title: "Automatic Differentiation",
                why: "MLPL's grad(loss, wrt) is backprop. This lesson shows the lift from hand-rolled chain rule to automatic differentiation.",
            },
            // 5. CNN (1989)
            Step::Note {
                title: "1989 -- Convolutional Neural Networks",
                body: "Weight sharing and translation invariance for image recognition (LeCun, 1989). LeNet-5 read handwritten zip codes at the US Postal Service. The same idea -- small learned filters sliding across the input -- is still the foundation of computer vision. Wider adoption: 2012 (AlexNet). Refinements: VGGNet (2014), Inception (2014), ResNet (2015).",
            },
            Step::Demo {
                name: "CNN (simple)",
                why: "A minimal conv2d -> relu -> pool2d pipeline. MLPL ships these builtins (saga 39).",
            },
            // 6. Universal approximation (1989)
            Step::Note {
                title: "1989 -- Universal approximation theorem",
                body: "A single hidden layer with enough neurons can approximate any continuous function to arbitrary precision (Cybenko 1989, Hornik 1991). The theorem guarantees existence but says nothing about learnability -- finding the right weights is the hard part. It explains why even shallow networks are powerful, and why depth helps (exponentially fewer neurons needed).",
            },
            // 7. LSTM (1997)
            Step::Note {
                title: "1997 -- LSTM",
                body: "Long Short-Term Memory: gated recurrence that solves the vanishing gradient problem (Hochreiter and Schmidhuber, 1997). Four gates (input, forget, output, cell) control what information to keep, add, or expose. Dominated sequence modeling from 1997 to 2017. Wider adoption: speech recognition (2013-2015), machine translation (2014-2016). Refinements: GRU (Cho et al. 2014), bidirectional LSTM, peephole connections.",
            },
            Step::Demo {
                name: "LSTM (sequence memory)",
                why: "A 10-step unrolled LSTM showing cell state persistence. MLPL ships lstm_cell (saga 41).",
            },
            // 8. AlexNet (2012)
            Step::Note {
                title: "2012 -- AlexNet and the GPU revolution",
                body: "Deep CNN trained on GPUs wins ImageNet by a huge margin (Krizhevsky, Sutskever, Hinton, 2012). Error rate dropped from 26% to 16% in one year. Proved that deep learning + big data + GPU compute was the winning formula. Triggered the modern deep learning era. Every major tech company pivoted to neural nets within 2 years.",
            },
            // 9. Word2Vec (2013)
            Step::Note {
                title: "2013 -- Word2Vec and dense embeddings",
                body: "Words as dense vectors where distance encodes meaning (Mikolov et al., 2013). 'king - man + woman = queen' captured the public imagination. Replaced sparse one-hot encodings with learned representations. Foundation for all modern NLP. MLPL ships embed_table for learned embeddings. Refinements: GloVe (2014), FastText (2016), contextual embeddings (ELMo 2018).",
            },
            // 10. VAE (2013)
            Step::Note {
                title: "2013 -- Variational Autoencoders",
                body: "Learned latent spaces with principled sampling (Kingma and Welling, 2013). An autoencoder whose bottleneck is a probability distribution, trained with a reconstruction loss plus a KL divergence regularizer. Enables generation by sampling from the latent space. Foundation for later diffusion models. MLPL ships a basic autoencoder demo; VAE-specific KL loss is deferred.",
            },
            Step::Glossary {
                term: "VAE (Variational Autoencoder)",
                why: "The probabilistic extension of autoencoders. Latent space is a distribution, not a point.",
            },
            // 11. GANs (2014)
            Step::Note {
                title: "2014 -- Generative Adversarial Networks",
                body: "Two networks in competition: Generator creates fakes, Discriminator catches them (Goodfellow et al., 2014). At Nash equilibrium the Generator produces data indistinguishable from real. Notoriously hard to train (mode collapse, training instability). Refinements: DCGAN (2015), WGAN (2017), StyleGAN (2018), BigGAN (2018). Largely superseded by diffusion models for image generation (2020+).",
            },
            Step::Demo {
                name: "GAN (2D circle)",
                why: "Generator learns to produce unit-circle points from noise. 50 alternating adam steps.",
            },
            // 12. Adam (2014)
            Step::Note {
                title: "2014 -- Adam optimizer",
                body: "Per-parameter adaptive learning rates with momentum and RMSProp combined (Kingma and Ba, 2014). The default optimizer for most deep learning since ~2016. First moment (momentum) tracks gradient direction; second moment (RMSProp) tracks gradient magnitude. Bias correction handles cold-start. Refinements: AdamW (Loshchilov, Hutter 2017) decouples weight decay from the adaptive step.",
            },
            Step::Glossary {
                term: "Adam",
                why: "MLPL's primary optimizer: adam(loss, params, lr, b1, b2, eps).",
            },
            // 13. Dropout (2014)
            Step::Note {
                title: "2014 -- Dropout",
                body: "Regularization by randomly zeroing activations during training (Srivastava et al., 2014, based on Hinton et al. 2012 idea). Forces the network to learn redundant representations -- no single neuron can be essential. At test time, all neurons fire but are scaled down. Simple, effective, still widely used. MLPL does not ship dropout; models are small enough that overfitting is not the bottleneck.",
            },
            Step::Glossary {
                term: "Dropout",
                why: "Random neuron silencing during training. A regularization technique.",
            },
            // 14. Batch Normalization (2015)
            Step::Note {
                title: "2015 -- Batch Normalization",
                body: "Normalize activations per mini-batch to stabilize training (Ioffe and Szegedy, 2015). Reduces internal covariate shift -- each layer sees inputs with stable statistics. Enables much higher learning rates and faster convergence. Nearly universal in CNNs and feedforward nets. Later alternatives: Layer Norm (2016, used in transformers), Group Norm (2018), RMS Norm. MLPL ships rms_norm for transformers.",
            },
            Step::Glossary {
                term: "Batch Normalization",
                why: "Per-batch mean/variance normalization. MLPL ships rms_norm (the transformer variant).",
            },
            // 15. ResNet (2015)
            Step::Note {
                title: "2015 -- ResNet and skip connections",
                body: "y = x + f(x): the residual connection (He et al., 2015). Solved the degradation problem -- deeper networks were performing worse than shallower ones. Skip connections let gradients flow directly through the identity path. Won ImageNet 2015 with 152 layers. Now universal: every transformer block uses residual connections. MLPL: residual(inner) in the model DSL.",
            },
            Step::Glossary {
                term: "Residual",
                why: "y = x + f(x). MLPL's residual(inner) wraps any layer with a skip connection.",
            },
            // 16. Attention / Transformer (2017)
            Step::Note {
                title: "2017 -- Attention Is All You Need",
                body: "The transformer: self-attention replaces recurrence entirely (Vaswani et al., 2017). Parallel computation over all positions, no sequential bottleneck. Scaled dot-product attention: softmax(QK^T/sqrt(d_k))V. Multi-head attention runs h parallel heads on d_k slabs. Changed everything: NLP (2018+), vision (2020+), protein folding (2020), audio (2022). MLPL ships attention, causal_attention, the full encoder/decoder block stack.",
            },
            Step::Lesson {
                title: "Self-Attention from Scratch",
                why: "Build one head of attention from primitives. The transformer's core in 15 lines.",
            },
            Step::Demo {
                name: "Tiny LM Generate",
                why: "End-to-end: BPE tokenizer + 1-layer transformer trained and sampled. The smallest program that learns to talk.",
            },
            // 17. GPT (2018)
            Step::Note {
                title: "2018 -- GPT: decoder-only language models",
                body: "Unsupervised pre-training + supervised fine-tuning on a decoder-only transformer (Radford et al., 2018). GPT-1 showed that a single architecture pre-trained on raw text could be fine-tuned for many tasks. GPT-2 (2019) showed emergent zero-shot abilities. GPT-3 (2020) showed in-context learning. GPT-4 (2023) is multimodal. The Tiny LM demos build a GPT-style stack.",
            },
            Step::Glossary {
                term: "GPT",
                why: "Decoder-only transformer family. MLPL's Tiny LM is a 1-layer GPT.",
            },
            // 18. BERT (2018)
            Step::Note {
                title: "2018 -- BERT: bidirectional pre-training",
                body: "Masked language modeling on an encoder-only transformer (Devlin et al., 2018). BERT reads both directions at once (unlike GPT's left-to-right). Dominated NLP benchmarks 2018-2020. Key insight: pre-train a deep bidirectional model, then fine-tune with one output layer for any task. Refinements: RoBERTa (2019), ALBERT (2019), DeBERTa (2020). MLPL has the encoder block but no masked-LM training loop.",
            },
            Step::Glossary {
                term: "BERT",
                why: "Encoder-only transformer with masked language model pre-training.",
            },
            // 19. Vision Transformer (2020)
            Step::Note {
                title: "2020 -- Vision Transformer (ViT)",
                body: "Image patches as tokens: apply the transformer directly to vision (Dosovitskiy et al., 2020). Patchify the image into a sequence, add positional embeddings, run through transformer encoder blocks. Matches or beats CNNs at scale. Key insight: attention over patches is all you need -- no convolutions. MLPL ships patchify + attention + trained ViT demos on Oxford Pets.",
            },
            Step::Demo {
                name: "ViT Attention Pattern (no training)",
                why: "The full ViT forward pipeline on a synthetic image, untrained. Shows every builtin composing into the recipe.",
            },
            // 20. Diffusion Models (2020)
            Step::Note {
                title: "2020 -- Diffusion models",
                body: "Learn to reverse a noise process: start with pure noise, iteratively denoise toward a data sample (Ho et al., DDPM 2020). Training is simple (predict the noise added at each step), generation is slow (many denoising steps). Foundation for Stable Diffusion (2022), DALL-E 2 (2022), and Midjourney. Wider adoption: 2022 (text-to-image explosion). Largely replaced GANs for image generation. MLPL does not ship diffusion primitives.",
            },
            // 21. RAG (2020)
            Step::Note {
                title: "2020 -- Retrieval-Augmented Generation",
                body: "Combine a retriever (search over a document corpus) with a generator (LLM) so the model can cite external knowledge instead of relying on memorized parameters (Lewis et al., 2020). The retriever finds relevant passages; the generator conditions on them. Foundation for grounded AI assistants, enterprise search, and reducing hallucination. Wider adoption: 2023+ (every major LLM product). MLPL has a glossary entry but no retrieval primitives.",
            },
            // 22. In-Context Learning (2020)
            Step::Note {
                title: "2020 -- In-Context Learning",
                body: "An emergent capability of large language models: provide examples in the prompt and the model learns the pattern without weight updates (Brown et al., GPT-3 2020). Zero-shot (no examples), few-shot (a handful), and many-shot variants. Not explicitly trained for -- it emerges from pre-training on diverse text at sufficient scale. Changed how practitioners use LLMs: prompt engineering replaced fine-tuning for many tasks.",
            },
            // 23. CLIP (2021)
            Step::Note {
                title: "2021 -- CLIP: connecting vision and language",
                body: "Dual-encoder contrastive learning: an image encoder and a text encoder trained to agree on matching pairs (Radford et al., 2021). Zero-shot image classification by comparing image embeddings to text-label embeddings. Foundation for text-to-image models (DALL-E, Stable Diffusion). MLPL has the attention primitives but not the dual-encoder training loop.",
            },
            Step::Glossary {
                term: "CLIP (Contrastive Language-Image Pre-training)",
                why: "Dual-encoder image+text contrastive model. Zero-shot classification via embedding similarity.",
            },
            // 24. LoRA (2021)
            Step::Note {
                title: "2021 -- LoRA: efficient fine-tuning",
                body: "Low-rank adapters: freeze the base model, train only small rank-r matrices injected at each layer (Hu et al., 2021). Reduces trainable parameters by 10,000x while matching full fine-tuning quality. Enables fine-tuning large models on consumer hardware. Refinements: QLoRA (2023) adds 4-bit quantization. MLPL ships lora(model, rank, seed) in the model DSL.",
            },
            Step::Lesson {
                title: "LoRA Fine-Tuning",
                why: "The MLPL lesson that demonstrates LoRA: freeze base weights, train rank-r adapters, compare loss curves.",
            },
            // 25. Chain-of-Thought (2022)
            Step::Note {
                title: "2022 -- Chain-of-Thought prompting",
                body: "Adding 'Let's think step by step' to a prompt dramatically improves LLM reasoning on math and logic problems (Wei et al., 2022). The model generates intermediate reasoning steps before the final answer. Foundation for inference-time compute scaling: more thinking tokens = better answers. Refinements: Tree-of-Thought (2023), self-consistency (sample multiple chains, take majority vote).",
            },
            // 26. RLHF + Constitutional AI (2022)
            Step::Note {
                title: "2022 -- RLHF and Constitutional AI",
                body: "Reinforcement Learning from Human Feedback: SFT -> reward model -> PPO policy optimization (Ouyang et al., 2022). The alignment technique behind ChatGPT. Constitutional AI (Bai et al., Anthropic 2022) replaces human labelers with AI feedback guided by a written constitution. Refinements: DPO (Rafailov et al., 2023) removes the reward model entirely, training directly on preference pairs.",
            },
            Step::Glossary {
                term: "RLHF (Reinforcement Learning from Human Feedback)",
                why: "The three-stage alignment pipeline: SFT, reward model, PPO. How ChatGPT was trained.",
            },
            // 27. Mixture of Experts (2022)
            Step::Note {
                title: "2017/2022 -- Mixture of Experts",
                body: "Route each token to k-of-N expert sub-networks (Shazeer et al. 2017; Fedus et al. 2022 scaled it). Each token uses only a fraction of the model's parameters, so compute cost scales sub-linearly with parameter count. GPT-4 and Mixtral are rumored/confirmed MoE architectures. Key challenge: load balancing across experts. MLPL does not ship MoE primitives.",
            },
            Step::Glossary {
                term: "MoE (Mixture of Experts)",
                why: "k-of-N routed experts per FFN. Compute scales sub-linearly with parameters.",
            },
            // 28. JEPA (2023)
            Step::Note {
                title: "2023 -- JEPA: Joint Embedding Predictive Architecture",
                body: "Predict missing information in embedding space rather than pixel space (Assran et al., I-JEPA 2023; framework by LeCun). Instead of reconstructing masked patches (like MAE), predict their latent representations. Avoids the shortcut problem where models learn to copy low-level textures. Part of LeCun's vision for a 'world model' that learns structured representations without generative reconstruction. Active research area.",
            },
            // 29. Mamba / SSM (2023)
            Step::Note {
                title: "2023 -- Mamba and state-space models",
                body: "Selective state-space models: an alternative to attention that runs in O(n) instead of O(n^2) (Gu and Dao, 2023). Mamba adds input-dependent selection to the S4 framework (Gu et al. 2021), achieving transformer-quality on language tasks with linear scaling. Whether SSMs fully replace transformers is an open question. MLPL does not ship SSM primitives.",
            },
            Step::Glossary {
                term: "State Space Models / Mamba",
                why: "Selective state-space alternative to attention. O(n) sequence processing.",
            },
            // 30. HRM + TRM (2025)
            Step::Note {
                title: "2025 -- Hierarchical and Tiny Recursive Models",
                body: "Two papers that challenge the scaling paradigm. HRM (Wang et al., 2025): a 27M-parameter recurrent architecture inspired by hierarchical brain processing achieves near-perfect Sudoku/maze results and strong ARC scores with only 1000 training samples. TRM (Jolicoeur-Martineau, 2025): simplifies HRM to a single 2-layer 7M-parameter network that recurses over its own output, scoring 45% on ARC-AGI-1 -- outperforming DeepSeek R1, o3-mini, and Gemini 2.5 Pro with less than 0.01% of their parameters. Key insight: recursive computation depth, not model width, drives generalization on abstract reasoning.",
            },
            // 31. RLM (2025)
            Step::Note {
                title: "2025 -- Recursive Language Models",
                body: "An inference paradigm where an LLM treats long prompts as an external environment, programmatically decomposing and recursively calling itself over prompt snippets (Zhang, Kraska, Khattab, 2025). Processes inputs up to 100x beyond the model's context window while outperforming vanilla frontier models. Reframes long-context processing as recursive self-invocation rather than context-window expansion. RLM-Qwen3-8B approaches GPT-5 quality on long-context tasks.",
            },
            // 32. ICRL (2026)
            Step::Note {
                title: "2026 -- In-Context Reinforcement Learning for tool use",
                body: "An RL-only framework that eliminates supervised fine-tuning by using few-shot in-context examples during RL rollouts, gradually reducing examples as the model learns independent tool use (Ye, Zhao, Duan et al., 2026). Achieves state-of-the-art on reasoning and tool-use benchmarks. Signals a shift: RL alone, with the right scaffolding, can teach complex behaviors that previously required curated SFT datasets.",
            },
            Step::Note {
                title: "The emerging pattern",
                body: "The 2024-2026 papers share a theme: depth of reasoning matters more than width of parameters. HRM and TRM show that tiny recursive models outperform billion-parameter LLMs on abstract reasoning. RLM shows that recursive self-invocation beats longer context windows. ICRL shows that RL scaffolding can replace supervised data. Chain-of-Thought showed that more thinking tokens improve answers. The frontier is shifting from 'how big is the model' to 'how deeply does it think'.",
            },
        ],
    },
    LearningPath {
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
    },
    LearningPath {
        title: "Build a transformer from primitives",
        blurb: "The from-scratch attention bundle: see the diagram, then build the same thing in MLPL with no model-DSL shortcuts. Nine steps; alternates diagrams and lessons.",
        steps: &[
            Step::Diagram {
                slug: "12_attention",
                why: "Single-head self-attention as one diagram. Frame the math before the code.",
            },
            Step::Lesson {
                title: "Self-Attention from Scratch",
                why: "Build one head: three projections (Q/K/V) + scaled dot-product + softmax + weighted sum. Every line has a paragraph hover tooltip.",
            },
            Step::Diagram {
                slug: "13_multi_head_attention",
                why: "How `h` heads run in parallel on `d_k = d_model/h` slabs.",
            },
            Step::Lesson {
                title: "Multi-Head Attention from Scratch",
                why: "Build the multi-head version using selector matrices for column slicing -- MLPL has no surface slice op, so the slabbing is explicit.",
            },
            Step::Lesson {
                title: "Cross-Attention from Scratch",
                why: "Same formula, but Q comes from a target sequence and K/V come from a separate source. Non-square weight heatmap is the visual signature.",
            },
            Step::Diagram {
                slug: "14_transformer_encoder",
                why: "How attention layers compose with FFN + residuals into a stackable encoder block.",
            },
            Step::Lesson {
                title: "Encoder Block",
                why: "Build one encoder block via the model DSL: chain(residual(rms_norm + self-attn), residual(rms_norm + ffn)).",
            },
            Step::Diagram {
                slug: "15_transformer_decoder",
                why: "Decoder = encoder + cross-attention sub-block. The third sub-block is the only difference.",
            },
            Step::Lesson {
                title: "Decoder Block",
                why: "Build the full three-sub-block decoder: causal self-attn + cross-attn (from scratch) + FFN. After this you have all the pieces of a real transformer.",
            },
        ],
    },
    LearningPath {
        title: "Data & Exploration",
        blurb: "Before you model, explore. This path walks the data side of ML: creating arrays, uploading images, inspecting shapes, visualizing distributions, generating synthetic datasets, and preparing data for training. Every step produces a picture or a number -- no models, no gradients, just getting to know your data.",
        steps: &[
            Step::Note {
                title: "Why exploration matters",
                body: "Most ML failures are data failures. A model trained on skewed data learns skewed patterns. A model trained on the wrong scale diverges. Spending 10 minutes looking at histograms and scatter plots before training saves hours of debugging after. This path builds the habits.",
            },
            Step::Lesson {
                title: "Hello Numbers",
                why: "Scalars, operators, the REPL. The absolute starting point -- everything else is arrays of numbers.",
            },
            Step::Lesson {
                title: "Arrays",
                why: "Vectors and their shapes. range(n) generates a sequence; reshape changes the layout. Arrays are the container for every dataset.",
            },
            Step::Lesson {
                title: "Matrices",
                why: "Reshape, transpose, slicing with take. A dataset is a matrix: rows are samples, columns are features.",
            },
            Step::Demo {
                name: "Basics",
                why: "Scalar arithmetic, elementwise ops, broadcasting, variable binding. The five-minute tour.",
            },
            Step::Demo {
                name: "Math Functions",
                why: "exp, log, sqrt, abs, sin, cos, sigmoid, tanh. The elementwise toolkit you will use for feature engineering and activation functions.",
            },
            Step::Note {
                title: "Generating synthetic data",
                body: "MLPL ships several synthetic dataset generators: blobs(seed, n, centers) for Gaussian clusters, moons(seed, n, noise) for two interleaving arcs, circles(seed, n, noise) for concentric rings, and random/randn for uniform/normal noise. Each returns a matrix ready for plotting or training. Seeded for reproducibility.",
            },
            Step::Demo {
                name: "Matrix Ops",
                why: "Reshape, transpose, matmul, dot. The shape-manipulation toolkit for arranging data into the format a model expects.",
            },
            Step::Lesson {
                title: "Visualizing Data",
                why: "svg(data, type) renders inline: line, bar, heatmap, scatter. One function, many views.",
            },
            Step::Demo {
                name: "Visualizations",
                why: "Line plots, bar charts, heatmaps in one line each. The visual vocabulary for data exploration.",
            },
            Step::Demo {
                name: "Analysis Helpers",
                why: "hist, scatter_labeled, loss_curve, confusion_matrix, boundary_2d. Higher-level plots that answer specific questions about your data or model.",
            },
            Step::Demo {
                name: "Upload & Inspect Image",
                why: "Bring your own data: :upload, check is_ok, inspect shape/mean/min/max, render with svg gallery, histogram of pixel intensities.",
            },
            Step::Glossary {
                term: ":upload (REPL command)",
                why: "The browser file picker: pick a photo, get a Result with pixels, height, width.",
            },
            Step::Lesson {
                title: "Loading Data",
                why: "load(path) reads CSV or text files. The terminal REPL needs --data-dir; the web playground has load_preloaded for bundled datasets.",
            },
            Step::Lesson {
                title: "Named Axes",
                why: "label(x, names) attaches semantic names to dimensions. 'batch', 'features', 'time' -- makes shapes self-documenting.",
            },
            Step::Note {
                title: "From exploration to modeling",
                body: "You now know how to create, load, inspect, and visualize data in MLPL. The next step is modeling: the 'Zero to LLM' path starts with logistic regression and builds to transformers. The 'Architecture Zoo' path surveys CNN, RNN, GAN, and attention side by side. Pick the one that matches your curiosity.",
            },
        ],
    },
    LearningPath {
        title: "Dimensionality reduction",
        blurb: "When the data lives in 50 dimensions but the screen has 2: pick a projection. PCA (linear, fast), t-SNE (local-only, dramatic), UMAP (local + global, the modern default). Six tutorial lessons in dependency order, three side-by-side demos at the end.",
        steps: &[
            Step::Lesson {
                title: "Why reduce dimensions?",
                why: "Concept-first motivation: the manifold hypothesis, the curse of dimensionality, why a screen has two axes but a learned embedding has 768. Frame the whole path.",
            },
            Step::Glossary {
                term: "Dimensionality reduction",
                why: "Reference card for the rest of the path: linear vs manifold methods, what each preserves, the MLPL builtins.",
            },
            Step::Demo {
                name: "PCA",
                why: "The cheap linear baseline. Power iteration on the covariance matrix finds the top axis of variance; project the points onto it. Linear, fast, deterministic.",
            },
            Step::Lesson {
                title: "PCA: the linear baseline",
                why: "Goes beyond the demo: pca() vs pca_components() vs pca_variance_explained(), the loadings-vs-projections distinction, when PCA's linear assumption is enough.",
            },
            Step::Glossary {
                term: "PCA (Principal Component Analysis)",
                why: "What PCA actually does (eigenvectors of the covariance matrix), what it misses (non-linear structure).",
            },
            Step::Demo {
                name: "PCA 3D (interactive)",
                why: "Same dataset, but project to 3 components and view in an interactive Plotly viewer. Drag/rotate confirms that well-separated 5-D clusters stay separated along every axis -- harder to fake than a single 2-D shot.",
            },
            Step::Demo {
                name: "PCA loadings (critical dimensions)",
                why: "Switch from 'where did the points go?' to 'which input dimensions matter?'. The critical-dimensions heatmap shows which features each PC is built from.",
            },
            Step::Lesson {
                title: "SNE: the very-slow ancestor",
                why: "t-SNE's predecessor (Hinton + Roweis 2002). Two failure modes -- asymmetric KL and the crowding problem -- set up exactly the two fixes t-SNE makes.",
            },
            Step::Lesson {
                title: "t-SNE: a peek at nonlinear methods",
                why: "How van der Maaten + Hinton fixed SNE: symmetric P + Student-t low-D affinity. Plus the 'cluster shape is meaningful, distance between clusters is not' caveat.",
            },
            Step::Glossary {
                term: "t-SNE",
                why: "Reference card: perplexity, KL, Student-t, why global distance is noise.",
            },
            Step::Lesson {
                title: "UMAP: the modern default",
                why: "Headline lesson. Riemannian-geometry framing, fuzzy simplicial sets, cross-entropy + negative sampling. Why UMAP preserves both local AND global structure where t-SNE preserves only local.",
            },
            Step::Glossary {
                term: "UMAP",
                why: "Reference card for the lesson: the math vocabulary in one place.",
            },
            Step::Demo {
                name: "UMAP vs PCA",
                why: "Two-moons embedded in 5-D. PCA reads the linear projection; UMAP reads the local k-NN graph. Both recover the moon arcs but via different recipes.",
            },
            Step::Demo {
                name: "UMAP vs t-SNE",
                why: "Three clusters where C is 5x farther than A is from B. t-SNE inflates every cluster to similar size; UMAP preserves the relative distance. This is the case the milestone is built around.",
            },
            Step::Glossary {
                term: "Multidimensional Scaling",
                why: "What MDS preserves (pairwise distances) and how it differs from PCA (variance directions) and t-SNE/UMAP (local neighborhoods). Background for the next demo.",
            },
            Step::Glossary {
                term: "Johnson-Lindenstrauss Lemma",
                why: "The sanity-baseline argument: a Gaussian random matrix preserves pairwise distances within (1 +- eps) for modest k. If a learned method does not beat random projection, the learned features are not adding signal.",
            },
            Step::Demo {
                name: "Dim-reduction zoo",
                why: "Same dataset, FIVE side-by-side projections (PCA / t-SNE / UMAP / MDS / random projection). The fastest way to internalize what each method emphasizes -- variance directions vs local neighborhoods vs pairwise distances vs the JL sanity baseline.",
            },
            Step::Lesson {
                title: "Reading a critical-dimensions heatmap",
                why: "Viz literacy. How to read the [k, D] critical-dimensions heatmap for PCA loadings; same conventions apply to upcoming permutation-sensitivity heatmaps for t-SNE / UMAP.",
            },
            Step::Note {
                title: "What's next",
                body: "The dim-reduction milestone is feature-complete after step 041 (MDS + random projection landed and joined the zoo). The remaining headroom is permutation-sensitivity heatmaps for t-SNE / UMAP (referenced in the 'Reading a critical-dimensions heatmap' lesson) -- those need a small helper builtin and ship as a follow-on saga. Track the milestone status in `docs/milestone-dimensionality-reduction.md`.",
            },
        ],
    },
    LearningPath {
        title: "REPL to Script",
        blurb: "Graduate from one-line REPL exploration to multi-line scripts to saved .mlpl files. Learn the editor, the save/load workflow, terminal script mode, user-defined functions, and script arguments. Eight steps; assumes you have run a few REPL expressions already.",
        steps: &[
            Step::Note {
                title: "From exploration to automation",
                body: "The REPL is for exploring: try an expression, see the result, adjust. Once you have something that works, you want to save it, run it again, and share it. This path walks the progression from interactive one-liners to reusable scripts.",
            },
            Step::Lesson {
                title: "Hello Numbers",
                why: "The REPL basics: type an expression, press Enter, see the result. Variables persist across lines. :vars shows what is bound.",
            },
            Step::Lesson {
                title: "Variables",
                why: "Name your intermediate values. x = range(10) binds an array; x persists until :clear. Variable names are your working memory.",
            },
            Step::Demo {
                name: "Workspace Introspection",
                why: "The REPL's self-awareness: :vars, :describe, :models, :fns, :wsid. Know what is in your session before you save it.",
            },
            Step::Note {
                title: "The Editor tab",
                body: "Click the Editor tab to get a multi-line text area. Type or paste several lines of MLPL, then press Ctrl+Enter (or the Run button) to execute them all. Output appears in the REPL pane below. The editor is a scratchpad -- it does not save automatically.",
            },
            Step::Note {
                title: "Saving and loading scripts",
                body: "The Save button downloads your editor content as a .mlpl file. The Load button opens a file picker to load a .mlpl file into the editor. The browser does not persist state between sessions -- save early, save often. You can also copy/paste between the editor and any text editor on your machine.",
            },
            Step::Demo {
                name: "User-Defined Functions",
                why: "def u:name(args) { body } defines a reusable function. This is how scripts become libraries: define your functions at the top, call them below.",
            },
            Step::Note {
                title: "Running scripts from the terminal",
                body: "The terminal REPL runs .mlpl files directly: mlpl-repl -f my_script.mlpl. Use -f (not stdin piping) because piping splits multi-line blocks like repeat {} across lines. Script arguments: mlpl-repl -f script.mlpl -- arg1 arg2. Inside the script, args() returns a string list and list_get(args(), 0) extracts one argument.",
            },
            Step::Note {
                title: "What comes next",
                body: "You now know the full REPL-to-script workflow: explore interactively, draft in the editor, save as .mlpl, run from the terminal with arguments. User-defined functions (def u:name) let you build reusable libraries. The Architecture Zoo and Zero to LLM paths show what to build with these tools.",
            },
        ],
    },
    LearningPath {
        title: "Vision Transformers in MLPL",
        blurb: "From a synthetic image to a trained multi-head cat-vs-dog classifier with per-head attention visualization, ending with uploading a photo from your phone for the trained model to classify. The same attention machinery from the transformer demos, applied to image patches. Alternates diagrams, glossary, and demos; assumes you have already seen scaled-dot-product attention.",
        steps: &[
            Step::Note {
                title: "How this path is laid out",
                body: "ViT is not new architecture -- it is the SAME [[Attention]] block from the transformer demos, fed image patches instead of token embeddings. The early steps explain the new pieces ([[patchify (builtin)]], [[take (builtin)]], batched and [[Multi-head attention]]); the demos wire them together at three depths: an untrained pattern, a trained single-head classifier with a labeled gallery, and a trained multi-head classifier with per-head attention maps. The path ends with bring-your-own-image (the [[:upload (REPL command)]] command) so you can classify a photo from your phone end-to-end. Click any underlined term to pop up its glossary entry.",
            },
            Step::Glossary {
                term: "patchify (builtin)",
                why: "ViT's one new primitive on top of standard transformer parts. Cuts an image into a sequence of flattened patches so attention can treat them as tokens.",
            },
            Step::Diagram {
                slug: "39_patchify",
                why: "How a 64x64 image becomes a [16, 768] token sequence. The first picture that ever made `patchify` click is usually this one.",
            },
            Step::Glossary {
                term: "take (builtin)",
                why: "Per-batch / per-row indexing. Used by the trained demo to extract single images from `pets_tiny.X` and to pull the first-token output of attention as a CLS-like aggregator.",
            },
            Step::Glossary {
                term: "Stack (tape op)",
                why: "Single-node, N-way concatenation along an existing axis. The multi-head autograd tape uses it to join per-head outputs (and the rank-3 path uses it to join per-batch outputs) in O(N) instead of an O(N^2) binary-concat chain.",
            },
            Step::Diagram {
                slug: "41_stack_tape_op",
                why: "Why we built Stack as a primitive instead of folding N concats. The before/after picture is the autograd cost story in one image.",
            },
            Step::Diagram {
                slug: "40_multi_head_attention",
                why: "How 4 heads share `d_model` and recombine. Pair this with the multi-head pattern demo below to read the [4, 17, 17] shape correctly.",
            },
            Step::Diagram {
                slug: "42_vit_pipeline",
                why: "The full forward path with shapes -- image -> patches -> embed + CLS + pos -> attention -> CLS -> MLP -> argmax. The map for everything that follows.",
            },
            Step::Demo {
                name: "ViT Attention Pattern (no training)",
                why: "Mechanical end-to-end forward pipeline on a synthetic image: patchify -> linear embed -> concat CLS -> + positional -> attention -> softmax -> heatmap. No training, so the heatmap is random; the demo's point is showing every Phase-1 builtin composing into the recipe.",
            },
            Step::Diagram {
                slug: "44_heatmap_grid",
                why: "How `svg(_, \"heatmap_grid\")` unfolds a [N, R, C] tensor into a grid. Required reading before the multi-head pattern demo, otherwise the four-cell layout is mystery geometry.",
            },
            Step::Demo {
                name: "ViT Multi-Head Attention Pattern (no training)",
                why: "Same pipeline but `attention(128, 4, ...)` -- four heads. attention_weights returns [4, 17, 17] and `svg(_, \"heatmap_grid\")` lays out a 2x2 grid of per-head heatmaps. Untrained, so all four look uniformly random and similar to each other; this is the baseline that the trained multi-head demo is judged against.",
            },
            Step::Glossary {
                term: "Oxford-IIIT Pet dataset",
                why: "The labeled image source. 7,393 cat/dog photos at ~500x500. MLPL ships a 200-image 64x64 subset (pets_tiny) embedded in the WASM binary so the trained demo runs without a server.",
            },
            Step::Demo {
                name: "Pets: cat vs dog (quick)",
                why: "Trained single-head Vision [[Transformer]] end-to-end: 8 balanced images, 30 adam steps, loss curve falls toward 0, training accuracy 1.0. The tail of the demo shows how to inspect the pets tensor with chained `take` calls.",
            },
            Step::Demo {
                name: "Pets: predict + gallery",
                why: "The same trained ViT, but now scaled to 16 images and run end-to-end with `predict_batch` + the 3-arg `svg(X, \"gallery\", overlay)` viz. Each thumbnail gets an `actual / predicted` caption -- you can finally look at a specific cat photo and read what the model said about it.",
            },
            Step::Demo {
                name: "Pets: multi-head ViT (quick + viz)",
                why: "The headline demo for the multi-head story. Same architecture as the single-head quick demo but `attention(128, 4)`; after 30 adam steps it renders the four post-training attention maps. Compare with the untrained multi-head pattern demo above to see what specialization gradient descent buys -- the heads start identical and end different.",
            },
            Step::Demo {
                name: "Pets: attention overlay (per-head)",
                why: "Same trained 4-head model, but the attention is rendered OVER the test image instead of as a [16, 16] heatmap. Bright yellow patches are what each head looks at; dark purple are ignored. The heatmap_grid tells you the [T, T] matrix per head; the overlay tells you WHERE on the image each head looks.",
            },
            Step::Glossary {
                term: ":upload (REPL command)",
                why: "Pick a photo from your device. The browser decodes + resizes it to 64x64 and binds the result under your chosen name as `Ok({pixels, h, w})`. A cancelled or unreadable upload binds `Err(\"...\")` instead so the program can branch on success without crashing on an undefined variable.",
            },
            Step::Diagram {
                slug: "43_result_type",
                why: "`Value::Result` and its accessors -- `is_ok`, `unwrap`, `err_message`, `unwrap_or`. The shape every `:upload` reply lands in.",
            },
            Step::Diagram {
                slug: "45_upload_result_flow",
                why: "End-to-end flow: REPL :upload -> file picker -> Canvas decode -> Ok([[Record]]) / Err(message). Read this before typing the five lines below.",
            },
            Step::Note {
                title: "Bring-your-own-image (try it now)",
                body: "After running any of the three trained pets demos above, the model bindings (`linear_p`, `attn`, `classifier`) stay in your REPL session. Then type these five lines (click any underlined term for its glossary entry):\n\n1. `:upload x` -- the [[:upload (REPL command)]] opens the file picker; pick any photo.\n2. `is_ok(x)` -- returns 1 if the upload succeeded, 0 if you dismissed the dialog or the file was not a valid image. (See the [[Result type]] for what `Ok` / `Err` mean.)\n3. `svg(unwrap(x).pixels, \"gallery\")` -- see the 64x64 resized version.\n4. `img = unwrap(x).pixels` -- pull the tensor out of the Result.\n5. `predict_batch(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(img, 16), [16, 768])), [1, 16, 128])), 1, 0))` -- returns `[0]` for cat or `[1]` for dog. Uses [[predict_batch (builtin)]] over a [[patchify (builtin)]]-and-attention forward pass.\n\nIf you dismissed the picker, `x = Err(\"cancelled\")` and `is_ok(x)` returns 0. Other Err flavors: `\"decode failed: not a valid image\"` (binary renamed as .jpg), `\"read failed\"` (zero-byte / permissions). Read the diagnostic with `err_message(x)`.\n\nReality check: the in-browser models train on only 8-20 images, so they memorize the training set and tend to classify any unseen photo as a cat. See `docs/better-cat-dog-future-demos.md` for the recommended improvement ladder (full-pets_tiny + val split, confidence threshold, augmentation, etc.).",
            },
            Step::Note {
                title: "Beyond this path",
                body: "**Full-resolution demo.** `demos/vit_multihead_thorough.mlpl` trains the same architecture on 128x128 input from the full Oxford-IIIT Pet (via `fetch_dataset`) inside a `device(\"mlx\")` block. Runs on Apple Silicon with `--features mlx`; falls back to CPU on any other host.\n\n**Pending architectural pieces.** [[Layer norm]] with learned affine and the tanh-approximation GELU are not yet builtins. Adding them would close the architectural gap to the upstream ViT notebook.\n\n**Making the classifier actually work on unseen photos.** Today's in-browser trained models overfit hard on 8-20 images and tend to classify every uploaded photo as a cat. The 7-demo improvement ladder in `docs/better-cat-dog-future-demos.md` lays out the recommended sequence: full-pets_tiny + held-out validation split first (no new builtins), then confidence-thresholded \"other\" output, then horizontal-flip augmentation, then [[AdamW]], then LayerNorm + GELU, then the thorough MLX run.",
            },
        ],
    },
    LearningPath {
        title: "Visual: ML by diagram",
        blurb: "Walk all 38 ML reference diagrams in numbered order. Pure browse path -- no MLPL code -- gives you the whole concept map before you dig into any single piece.",
        steps: &[
            Step::Note {
                title: "How to use this path",
                body: "Each step is one diagram. The blurb says what slice of MLPL covers the same ground (or notes that we have only the glossary entry, not a runnable demo). Skim the whole path first to get the lay of the land; come back to specific diagrams when you start a topic.",
            },
            Step::Diagram {
                slug: "01_linear_regression",
                why: "y = wX + b + MSE + gradient descent. The smallest-possible ML loop.",
            },
            Step::Diagram {
                slug: "02_logistic_regression",
                why: "Add a sigmoid + cross-entropy. Now it is a classifier. We have the demo + lesson.",
            },
            Step::Diagram {
                slug: "03_decision_tree",
                why: "Greedy yes/no splits on features. Glossary entry only -- no MLPL primitive.",
            },
            Step::Diagram {
                slug: "04_random_forest",
                why: "Bagged ensemble of decision trees. Glossary only.",
            },
            Step::Diagram {
                slug: "05_svm",
                why: "Maximum-margin hyperplane + kernel trick. Pre-deep-learning state of the art. Glossary only.",
            },
            Step::Diagram {
                slug: "06_perceptron",
                why: "Rosenblatt 1958 -- one linear layer + threshold. We use it in the History of ML lesson.",
            },
            Step::Diagram {
                slug: "07_mlp",
                why: "Stack linear + nonlinearity. Tiny MLP demo + lesson cover this.",
            },
            Step::Diagram {
                slug: "08_cnn",
                why: "Conv + pool + FC. MLPL ships conv2d and pool2d builtins (saga 39) plus a Simple CNN demo.",
            },
            Step::Diagram {
                slug: "09_resnet",
                why: "y = x + f(x). MLPL has `residual(...)` directly; the encoder/decoder block lessons use it.",
            },
            Step::Diagram {
                slug: "10_rnn",
                why: "Hidden state passed through time. MLPL ships rnn_cell (saga 41) plus an RNN sequence demo.",
            },
            Step::Diagram {
                slug: "11_lstm",
                why: "Gated RNN cell with four gates. MLPL ships lstm_cell (saga 41) plus an LSTM sequence demo.",
            },
            Step::Diagram {
                slug: "12_attention",
                why: "softmax(Q K^T / sqrt(d_k)) V. Self-Attention from Scratch lesson + [[Attention]] Pattern demo.",
            },
            Step::Diagram {
                slug: "13_multi_head_attention",
                why: "h heads on d_k slabs. Multi-Head Attention from Scratch lesson + demo.",
            },
            Step::Diagram {
                slug: "14_transformer_encoder",
                why: "Stack of encoder blocks. Encoder Block lesson + demo.",
            },
            Step::Diagram {
                slug: "15_transformer_decoder",
                why: "Causal self-attn + cross-attn + FFN. Decoder Block lesson + demo.",
            },
            Step::Diagram {
                slug: "16_encoder_decoder_transformer",
                why: "Full seq-to-seq. We have the parts (encoder, decoder); no end-to-end demo yet.",
            },
            Step::Diagram {
                slug: "17_gpt_decoder_only",
                why: "Tiny LM IS this: stacked causal-self-attn blocks.",
            },
            Step::Diagram {
                slug: "18_moe",
                why: "k-of-N routed experts per FFN. Glossary only.",
            },
            Step::Diagram {
                slug: "19_rag",
                why: "Retrieve docs, prepend, generate. Glossary only -- deferred.",
            },
            Step::Diagram {
                slug: "20_agent_loop",
                why: "LLM tool-use cycle. Glossary entry; `llm_call` is the building block.",
            },
            Step::Diagram {
                slug: "21_vit",
                why: "Patches as tokens. Glossary only -- needs image inputs.",
            },
            Step::Diagram {
                slug: "22_unet",
                why: "Conv encoder-decoder + skips. Glossary only.",
            },
            Step::Diagram {
                slug: "23_diffusion",
                why: "Iterative denoising. Glossary only -- deferred.",
            },
            Step::Diagram {
                slug: "24_clip",
                why: "Dual-encoder image + text. Glossary only -- needs image inputs.",
            },
            Step::Diagram {
                slug: "25_vlm",
                why: "Vision encoder + projector + LM. Glossary only.",
            },
            Step::Diagram {
                slug: "26_mamba_ssm",
                why: "Selective state-space alternative to attention. Glossary only -- deferred.",
            },
            Step::Diagram {
                slug: "27_training_loop",
                why: "Forward -> loss -> backward -> step. Every training demo IS this.",
            },
            Step::Diagram {
                slug: "28_backprop",
                why: "Reverse-mode chain rule. Why backprop? lesson + Automatic Differentiation lesson cover this.",
            },
            Step::Diagram {
                slug: "29_data_parallel_training",
                why: "Replicate model, split batch, all-reduce. Glossary only.",
            },
            Step::Diagram {
                slug: "30_tensor_parallel_training",
                why: "Split layer weights across devices. Glossary only.",
            },
            Step::Diagram {
                slug: "31_pipeline_parallel_training",
                why: "Split layers across devices. Glossary only.",
            },
            Step::Diagram {
                slug: "32_lora",
                why: "Low-rank adapters on a frozen base. LoRA Fine-Tuning lesson covers this.",
            },
            Step::Diagram {
                slug: "33_qlora",
                why: "Int4 base + bf16 LoRA. Glossary only -- LoRA exists, quantization does not.",
            },
            Step::Diagram {
                slug: "34_rlhf",
                why: "SFT -> reward model -> PPO. Glossary only -- deferred.",
            },
            Step::Diagram {
                slug: "35_dpo",
                why: "Direct preference optimization. Glossary only.",
            },
            Step::Diagram {
                slug: "36_self_play_training",
                why: "Agent generates its own training signal. Glossary only -- deferred.",
            },
            Step::Diagram {
                slug: "37_grokking",
                why: "Delayed generalization after long memorization. Glossary only -- a research curiosity.",
            },
            Step::Diagram {
                slug: "38_superposition",
                why: "Networks pack more features than dimensions. Glossary only -- mechanistic interpretability.",
            },
        ],
    },
    LearningPath {
        title: "Zero to LLM",
        blurb: "The spine: orientation -> arrays -> classifiers -> MLP -> autograd -> attention -> transformer -> tiny LM. Twelve steps; assume zero ML background.",
        steps: &[
            Step::Lesson {
                title: "What is ML, and why are we here?",
                why: "Set the destination first. Every later step is a variation of the same recipe (data, model, loss, gradient descent).",
            },
            Step::Lesson {
                title: "Hello Numbers",
                why: "MLPL's smallest possible expressions. Numbers and operators -- the substrate every other lesson is built on.",
            },
            Step::Lesson {
                title: "Arrays",
                why: "Vectors, then matrices. APL-derived shape semantics that ML inherits.",
            },
            Step::Lesson {
                title: "Matrices",
                why: "Reshape, transpose, dimension manipulation. The shape-arithmetic layer ML stands on.",
            },
            Step::Lesson {
                title: "Math and Activations",
                why: "exp, log, sigmoid, tanh -- the elementwise primitives every neural layer composes.",
            },
            Step::Lesson {
                title: "Machine Learning: Logistic Regression",
                why: "The hello-world ML model: fit two weights to four points using hand-rolled gradient descent. Forward + backward pass written out explicitly.",
            },
            Step::Lesson {
                title: "Going Non-Linear: A Tiny MLP",
                why: "Add a hidden layer + tanh. Solves problems no linear model can. The chain-rule backward pass is visible in the code.",
            },
            Step::Lesson {
                title: "Automatic Differentiation",
                why: "Replace the hand-rolled chain rule with `grad(loss, wrt)`. The lift from manual derivation to automatic differentiation that backprop unlocked in 1986.",
            },
            Step::Diagram {
                slug: "12_attention",
                why: "Visual reference for scaled-dot-product attention before reading the from-scratch implementation. The whole formula in one diagram.",
            },
            Step::Lesson {
                title: "Self-Attention from Scratch",
                why: "Build one head of attention from primitives -- three projections, score, softmax, weighted sum. The transformer's core in 15 lines.",
            },
            Step::Diagram {
                slug: "17_gpt_decoder_only",
                why: "Where a single attention layer fits in a stacked decoder-only transformer. Visualizes what \"Tiny LM\" actually instantiates.",
            },
            Step::Demo {
                name: "Tiny LM Generate",
                why: "End-to-end: BPE tokenizer + 1-layer transformer LM trained 30 steps on a tiny corpus, then sampled to generate text. The smallest program that learns to talk.",
            },
        ],
    },
];
