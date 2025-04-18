
# Model Calibration

To personalize the model, TumorTwin calibrates parameters to match observed data using:

\[
p^* = \arg\min_p L(p; o)
\]

where:
- \( p \): model parameters (e.g., \( k, D \))
- \( o \): observed cellularity maps
- \( L \): loss function (e.g., MSE between predicted and observed)

Gradients of \( L \) are computed via backpropagation or adjoint methods.

Supported optimizers include:
- PyTorch built-ins (e.g., LBFGS)
- Custom Levenberg-Marquardt (LM)
    