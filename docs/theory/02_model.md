
# Tumor Growth Model

TumorTwin implements a 3D reaction-diffusion model with treatment effects:

\[
\frac{\partial N(x, t)}{\partial t} = \nabla \cdot (D \nabla N) + k(x) N\left(1 - \frac{N}{\theta}\right)
- \sum_{i=1}^{n_{CT}} \sum_{j=1}^{T_i} \alpha_i C_i e^{-\beta_i(t - \tau_{i,j})} N
\]

Radiotherapy is applied as a discrete event:

\[
N^+ = N^- e^{-\alpha_{RT} d(t) - \beta_{RT} d(t)^2}
\]

## Discretization

The PDE is discretized using finite differences on a voxel grid, yielding:

\[
\frac{dN}{dt} = D L N + k N\left(1 - \frac{N}{\theta} \right) - \text{CT terms}
\]

Model parameters:
- \( D \): diffusion coefficient
- \( k \): proliferation rate
- \( \theta \): carrying capacity
- \( \alpha_i, \beta_i \): chemotherapy effects
- \( \alpha_{RT}, \beta_{RT} \): radiotherapy effects
    