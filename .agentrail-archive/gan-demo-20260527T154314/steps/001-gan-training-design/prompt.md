Step 001: GAN training loop design.

Investigate whether the existing train block + grad() can support alternating discriminator/generator updates. Design the GAN demo approach: generator (noise -> 2D points), discriminator (2D points -> real/fake score). Write a test that exercises the core GAN loop pattern. If new builtins are needed, implement them with TDD.