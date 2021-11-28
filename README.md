# flows
Implementations of a few normalizing flow models

## RealNVP
[Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) -- paper on arxiv

No multiscaling and ActNorm from [Glow](https://arxiv.org/abs/1807.03039) instead of batch normalization

Not conditional

![realnvp_mnist](generated/realnvp_mnist.png?raw=true)

## MAF

[Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057) -- paper on arxiv

Conditional

![maf_mnist](generated/maf_mnist.png?raw=true)
