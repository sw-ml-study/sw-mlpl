use crate::types::{LearningPath, Step};

pub(super) const PATH_A_CHRONOLOGICAL_HISTORY_OF_ML: LearningPath = LearningPath {
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
};
