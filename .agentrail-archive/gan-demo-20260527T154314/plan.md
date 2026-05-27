# GAN Demo

Build a simple GAN (Generative Adversarial Network) demo
that learns a 2D distribution. The generator maps random
noise to 2D points; the discriminator classifies real vs
fake. Train with alternating grad() updates using the
existing train block.

## Steps

1. GAN training loop design + discriminator/generator
   builtins or macros if needed
2. Simple GAN demo (2D circle distribution)
3. Glossary (GAN, Generator, Discriminator, Adversarial
   Training, Latent Space) + help text + language-status
   + saga close