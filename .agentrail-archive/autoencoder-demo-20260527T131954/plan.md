# ML Architecture Zoo Learning Path

## Unifying concept

"How Machines Learn to See, Generate, and Remember"

Each ML architecture exploits a different kind of structure
in data. This path walks a beginner through the major
families in dependency order, building intuition for WHEN
to reach for each tool.

| Architecture | Structure it exploits |
|-------------|----------------------|
| Dense/MLP | Fixed-size input -> fixed-size output |
| CNN | Spatial locality (nearby pixels matter most) |
| RNN/LSTM | Sequential order (what came before matters) |
| Autoencoder | Compression -> reconstruction (bottleneck) |
| GAN | Adversarial feedback (generator vs critic) |
| Transformer | Long-range attention (any-to-any) |

## Learning path: "From Pixels to Language"

### Group 1: See (spatial structure)

1. Note: "What is an architecture?"
2. Glossary: Neural network
3. Lesson: "Dense layers: the universal approximator"
4. Demo: Moons MLP (existing)
5. Lesson: "CNN: exploiting spatial structure"
6. Demo: Simple CNN (existing, saga 39)
7. Lesson: "Pooling and stride: controlled information loss"
8. Glossary: Convolution, Feature map, Pooling

### Group 2: Remember (sequential structure)

9. Lesson: "RNN: sequence memory"
10. Demo: Simple RNN (needs RNN builtins)
11. Lesson: "The vanishing gradient problem"
12. Lesson: "LSTM/GRU: gated memory"
13. Demo: LSTM on a sequence (needs LSTM builtins)
14. Glossary: RNN, LSTM, Vanishing gradient

### Group 3: Create (generative structure)

15. Lesson: "Autoencoders: learning to compress"
16. Demo: Simple autoencoder (needs demo)
17. Lesson: "GANs: learning by competition"
18. Demo: Simple GAN on 2D distributions (needs GAN loop)
19. Glossary: Autoencoder, GAN, Latent space

### Group 4: Attend (global structure)

20. Lesson: "Attention: the transformer insight"
21. Demo: Self-Attention from Scratch (existing)
22. Lesson: "Multi-head attention and positional encoding"
23. Demo: Multi-Head Attention (existing)
24. Lesson: "Why transformers replaced RNNs"
25. Glossary: Attention, Transformer, Positional encoding
26. Note: "Where to go from here"

## What exists vs what needs building

### Exists (no new builtins needed)

- Dense/MLP: Moons MLP, Tiny MLP, Logistic Regression demos
- CNN: conv2d, pool2d, relu, Simple CNN demo (saga 39)
- Attention/Transformer: Self-Attention, Multi-Head,
  Encoder Block, Decoder Block, ViT demos
- Autoencoder: can be built with existing model DSL
  (chain of linear layers with shrinking then expanding
  dimensions). Needs a demo, not new builtins.

### Needs new builtins

- **RNN cell**: `rnn_cell(input, hidden, W_ih, W_hh, bias)`
  -- one step of a simple recurrent cell. Returns new hidden
  state. Pure matrix ops: `tanh(W_ih @ input + W_hh @ hidden + bias)`.
  Could be implemented as a macro over existing ops, but a
  builtin makes the demo cleaner.

- **LSTM cell**: `lstm_cell(input, hidden, cell, W, bias)`
  -- one step of an LSTM. Returns (new_hidden, new_cell).
  Four gates (input, forget, cell, output) computed from
  concatenated input+hidden, split, sigmoided/tanhed.

- **Sequence unrolling**: `unroll(cell_fn, inputs, h0)` or
  a `for` loop over time steps. MLPL already has `for/in`
  loops (saga 31) so unrolling can be done in user code.

### Needs new framework

- **GAN training loop**: alternating generator and
  discriminator updates. The existing `train` block updates
  one loss; GANs need two losses alternating. Could be done
  with two explicit `train` blocks per iteration, or a new
  `train_gan` primitive.

## Suggested sagas

### Saga 40: Autoencoder demo

No new builtins needed. Build a demo using:
```
encoder = chain(linear(784, 128, seed), relu_layer(),
                linear(128, 32, seed2))
decoder = chain(linear(32, 128, seed3), relu_layer(),
                linear(128, 784, seed4))
autoencoder = chain(encoder, decoder)
```
Train on a simple synthetic dataset. The 3D viz shows
the bottleneck (32-element vector) vs the full input/output.

Steps: autoencoder demo, latent-space visualization,
reconstruction comparison, path entries, glossary.
~4 steps.

### Saga 41: RNN/LSTM builtins + demos

New builtins: `rnn_cell`, `lstm_cell`. Demos: character
prediction (predict next character from a short sequence),
simple time-series. The 3D viz shows hidden state evolving
over time steps.

Steps: rnn_cell builtin, lstm_cell builtin, char-predict
demo, time-series demo, 3D sequence viz, path entries.
~6 steps.

### Saga 42: GAN framework + demo

Extend the `train` block or add `train_gan` for alternating
updates. Demo: 2D GAN that learns a distribution (e.g.,
generate points on a circle from random noise). The 3D viz
shows the generator output distribution converging toward
the real distribution over training steps.

Steps: GAN training loop, simple 2D GAN demo, 3D
distribution viz, path entries, glossary.
~5 steps.

### Saga 43: Architecture Zoo path assembly

Assemble all the lessons, demos, and glossary entries into
the "From Pixels to Language" learning path. Write the
connector lessons (dense layers, vanishing gradient,
why transformers replaced RNNs). Add to the Paths tab.

Steps: 8 new lessons, path definition, tour stop, glossary
entries, close.
~5 steps.

## Quality requirements

Same as saga 39. TDD for builtins. Warning-target design.
Each saga independently shippable.
