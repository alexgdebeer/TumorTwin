
# Theory Overview: TumorTwin

TumorTwin is a Python framework for developing patient-specific digital twins in oncology, enabling predictive modeling of tumor growth and treatment response. This document provides a concise overview of the theoretical foundation implemented in the TumorTwin codebase.

## Core Model

At the heart of TumorTwin is a **reaction-diffusion tumor growth model** that incorporates both chemotherapy and radiotherapy effects. The governing equation for tumor cell density \( N(x, t) \) is given by:

\[
\frac{\partial N(x, t)}{\partial t} = \nabla \cdot (D \nabla N(x, t)) + k(x) N(x, t)\left(1 - \frac{N(x, t)}{\theta}\right)
- \sum_{i=1}^{n_{CT}} \sum_{j=1}^{T_i} \alpha_i C_i e^{-\beta_i (t - \tau_{i,j})} N(x, t)
\]

Radiotherapy is modeled as an instantaneous effect:

\[
N(x, t)^+ = N(x, t)^- \exp(-\alpha_{RT} d_{RT}(t) - \beta_{RT} d_{RT}(t)^2)
\]

where:
- \( D \): diffusion coefficient
- \( k(x) \): proliferation rate
- \( \theta \): carrying capacity
- \( \alpha_i, \beta_i \): chemotherapy efficacy and decay
- \( \alpha_{RT}, \beta_{RT} \): radiotherapy radiosensitivity parameters
- \( C_i \): normalized chemotherapy dose
- \( \tau_{i,j} \): administration times

## Discretization

The PDE is discretized spatially using a finite-difference method, aligning the computational grid with imaging voxel resolution. The result is a system of ODEs solved using:

\[
\frac{dN}{dt} = D L N + k N \left(1 - \frac{N}{\theta} \right) - \text{CT terms}
\]

where \( L \) is the discrete Laplace operator.

## Model Calibration

Model calibration involves fitting model parameters \( p = \{k, D\} \) to patient-specific data derived from MRI. The optimization problem is:

\[
p^* = \arg\min_p L(p; o)
\]

where \( o \) are cellularity maps inferred from ADC MRI using:

\[
N(x, t) = \frac{ADC_w - ADC(x, t)}{ADC(x, t) - ADC_{min}}
\]

Gradients of the loss \( L \) are computed via **adjoint-based automatic differentiation**, enabling memory-efficient optimization.

## Forward Solvers and Performance

Forward prediction of tumor growth is performed with differentiable ODE solvers from `torchdiffeq`, supporting:
- Fixed-step methods (e.g., Runge-Kutta 4)
- Adaptive-step methods (e.g., Dormand-Prince)

These solvers are compatible with CPU and GPU architectures, with ~10-15x speedups on GPU.

## Summary

TumorTwin implements a modular, high-performance framework for personalized tumor modeling, supporting:
- Reaction-diffusion models with treatment terms
- Voxel-based spatial discretization
- Differentiable solvers and efficient gradient computation
- Patient-specific calibration from imaging data

This theoretical backbone allows TumorTwin to serve as a flexible research tool for exploring modeling choices, treatment simulations, and decision support in oncology.

