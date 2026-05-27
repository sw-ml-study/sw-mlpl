Step 001: Autoencoder demo.

Build a demo using existing builtins -- no new runtime work needed. The autoencoder is two chains: encoder shrinks dimensions, decoder expands back.

encoder = chain(linear(input_dim, hidden, seed), relu_layer(), linear(hidden, latent, seed2))
decoder = chain(linear(latent, hidden, seed3), relu_layer(), linear(hidden, input_dim, seed4))

Use a simple synthetic dataset: random 16-element vectors. Train to minimize reconstruction error (MSE between input and output). Show the latent bottleneck vector (compressed representation).

New demos_autoencoder.rs with the demo. Register in DEMOS array with category 'Autoencoder'. Pages rebuild required.