import numpy as np
import pytest
from matplotlib import pyplot as plt

from tumortwin.preprocessing.bound_condition_maker import bound_condition_maker
from tumortwin.solvers.predict_cell_count import predict_cell_count
from tumortwin.types.imaging import NibabelNifti


def analytic_solution(t, U_0, k):
    return U_0 * np.exp(k * t) / (1 - U_0 + U_0 * np.exp(k * t))


def mae(u, v):
    return np.mean(np.abs(np.array(u) - np.array(v)))


@pytest.mark.parametrize(
    "U_0, k_0",
    [(0.5, 1.0), (0.5, 1.0), (0.5, 2.0), (0.7, 2.0), (0.1, 2.0), (0.1, 1.0)],
)
def test_reaction_model(U_0: float, k_0: float, plot=False):
    dt = 0.05
    ts = np.arange(0, 10 + dt, dt)

    theta = 1.0  # Maximum Cellularity

    D = 0.0

    domain_shape = (3, 3, 3)
    mask_array = np.ones(domain_shape)
    mask_array[0, :, :] = 0.0
    mask_array[-1, :, :] = 0
    mask_array[:, 0, :] = 0
    mask_array[:, -1, :] = 0
    mask_array[:, :, 0] = 0
    mask_array[:, :, -1] = 0

    cellularity_array = U_0 * mask_array
    k_array = k_0 * mask_array

    mask = NibabelNifti.from_array(mask_array)
    cellularity = NibabelNifti.from_array(cellularity_array)
    k = NibabelNifti.from_array(k_array)

    bcs = bound_condition_maker(mask)

    u_analytic = []
    for t in ts:
        u_analytic.append(analytic_solution(t, U_0, k_0))

    _, results, _ = predict_cell_count(
        t=(ts[0], ts[-1]),
        dt=dt,
        initial_cellularity_image=cellularity,
        d=D,
        theta=theta,
        k=k,
        bcs=bcs,
        brain_mask=mask,
    )
    u_computed = [r.array[1, 1, 1] for r in results]

    if plot:
        plt.plot(ts, u_analytic, "k--", label="analytic")
        plt.plot(ts, u_computed, label="computed")

    for t in ts:
        assert mae(u_computed, u_analytic) < 1e-2
