
# Gradient Computation

Gradients are used for model calibration and sensitivity analysis.

TumorTwin supports:
- Automatic differentiation (PyTorch)
- Memory-efficient adjoint-based gradients (`use_adjoint=True` in solver)

Example:
\[
\frac{\partial}{\partial k} TTC(200)
\]

where \( k \) is the proliferation rate and TTC is computed from model predictions.

This enables gradient-based optimization using PyTorch or custom solvers.
    