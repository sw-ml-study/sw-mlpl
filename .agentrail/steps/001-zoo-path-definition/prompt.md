Step 001: Architecture Zoo path definition.

Create the "From Pixels to Language" learning path in paths.rs with 4 groups:
1. See (spatial): CNN glossary + Simple CNN demo + pooling note
2. Remember (sequential): RNN glossary + RNN demo + vanishing gradient glossary + LSTM glossary + LSTM demo
3. Create (generative): Autoencoder glossary + Autoencoder demo + GAN glossary + GAN demo + Latent Space glossary
4. Attend (global): existing transformer path content (attention diagrams, self-attention lesson, Tiny LM demo)

Also update stale diagram path entries in the Visual path (CNN diagram now says "conv2d shipped", RNN/LSTM diagrams now say "rnn_cell/lstm_cell shipped"). Demo smoke test.