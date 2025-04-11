import matplotlib.pyplot as plt
import numpy as np
import pytest

from tumortwin.preprocessing.bound_condition_maker import bound_condition_maker
from tumortwin.solvers.predict_cell_count import predict_cell_count
from tumortwin.types.imaging import NibabelNifti


def analytic_solution(x, t, U0, L, D, num_terms=10000):
    """
    Calculate the analytic solution of the
    1D diffusion equation with zero-flux boundary conditions.

    Parameters:
    x (float or array-like): Spatial coordinate(s) where the solution is evaluated.
    t (float): Time at which the solution is evaluated.
    U0 (float): Initial concentration in the left half of the domain.
    L (float): Length of the domain.
    D (float): Diffusion coefficient.
    num_terms (int): Number of terms to include in the series solution.

    Returns:
    u (float or array-like): Concentration at position(s) x and time t.
    """
    x = np.asarray(x)
    u = np.zeros_like(x, dtype=float)

    for n in range(1, num_terms * 2, 2):  # odd n only
        A_n = (2 * U0 / (n * np.pi)) * (-1) ** ((n - 1) // 2)
        term = A_n * np.cos(n * np.pi * x / L) * np.exp(-D * (n * np.pi / L) ** 2 * t)
        u += term
    u += 0.5
    return u


def mae(u, v):
    return np.mean(np.abs(u - v))


@pytest.mark.parametrize("D, dt", [(0.1, 0.5), (1.0, 0.05), (10.0, 0.005)])
def test_diffusion_model(D: float, dt: float, plot=False):

    theta = 1.0  # Maximum Cellularity
    N = 101
    L = 100
    U_0 = 1.0
    k_0 = 0.0
    domain_shape = (N, 6, 6)

    cellularity_array = np.zeros(domain_shape)
    cellularity = NibabelNifti.from_array(cellularity_array)

    for x, y, z in np.ndindex(cellularity_array.shape):
        if x <= N / 2:
            cellularity_array[x, y, z] = U_0

    mask_array = np.ones(domain_shape)
    mask_array[0, :, :] = 0.0
    mask_array[-1, :, :] = 0
    mask_array[:, 0, :] = 0
    mask_array[:, -1, :] = 0
    mask_array[:, :, 0] = 0
    mask_array[:, :, -1] = 0
    mask = NibabelNifti.from_array(mask_array)

    k_array = k_0 * mask_array
    k = NibabelNifti.from_array(k_array)

    bcs = bound_condition_maker(mask)

    xs = np.linspace(0, L, N)  # Spatial points

    t_f = 500
    _, results, _ = predict_cell_count(
        t=(0, t_f),
        dt=dt,
        initial_cellularity_image=cellularity,
        d=D,
        theta=theta,
        k=k,
        bcs=bcs,
        brain_mask=mask,
    )

    for t in [10, 50, 100, 500]:
        u_analytic = analytic_solution(xs, t, U_0, L, D)
        result_idx = int(t // dt)
        u_comp = results[result_idx].array[:, 3, 3]
        if plot:
            print(mae(u_comp[1:-1], u_analytic[1:-1]))
            plt.plot(xs[1:-1], u_comp[1:-1], label=f"t = {t_f}")
            plt.plot(xs[1:-1], u_analytic[1:-1], "k--")
        assert mae(u_comp[1:-1], u_analytic[1:-1]) < 1e-2

    if plot:
        plt.legend()
        plt.show()
