
# Numerical Solvers

TumorTwin uses differentiable ODE solvers from `torchdiffeq` to simulate tumor dynamics.
Solvers include:
- Fixed-step: Runge-Kutta 4 (RK4)
- Adaptive-step: Dormand-Prince 5 (Dopri5)

Solvers integrate the ODEs from the discretized model efficiently on CPU or GPU.

Radiotherapy is handled via discrete events injected into the time integration loop.

## Total Tumor Cell Count

A key prediction metric is the total tumor cell count (TTC):

\[
TTC(t) = \sum_i N_i(t) \, \theta \, V_{voxel}
\]
    