import copy
from typing import List, Tuple

import numpy as np
import pytest
from matplotlib import pyplot as plt

from tumortwin.preprocessing.bound_condition_maker import bound_condition_maker
from tumortwin.solvers.predict_cell_count import predict_cell_count
from tumortwin.treatments.radiotherapy import compute_radiotherapy_cell_death_fractions
from tumortwin.types.imaging import NibabelNifti
from tumortwin.types.treatment import RadiotherapySpecification


def analytic_solution(
    N: float, times: List[float], radiotherapy_specification: RadiotherapySpecification
) -> Tuple[List[float], List[float]]:
    radiotherapy_times = np.array(list(radiotherapy_specification.protocol.keys()))
    solve_times = np.sort(
        np.concatenate((radiotherapy_times, np.union1d(radiotherapy_times, times)))
    )
    radiotherapy_effects = compute_radiotherapy_cell_death_fractions(
        radiotherapy_specification
    )

    N_out = [copy.deepcopy(N)]
    for t_0, t_1 in zip(solve_times[:-1], solve_times[1:]):
        if t_0 == t_1:
            N *= radiotherapy_effects[t_0]
        N_out.append(copy.deepcopy(N))

    return solve_times.tolist(), N_out


def mae(u, v):
    return np.mean(np.abs(np.array(u) - np.array(v)))


@pytest.mark.parametrize(
    "N_init, radiotherapy_specification",
    [
        (1.0, RadiotherapySpecification(alpha=1.0, protocol={1: 0.5})),
        (0.5, RadiotherapySpecification(alpha=1.0, protocol={1: 0.5})),
        (1.0, RadiotherapySpecification(alpha=1.0, protocol={0.5: 0.5})),
        (1.0, RadiotherapySpecification(alpha=0.5, protocol={1: 0.5})),
        (1.0, RadiotherapySpecification(alpha=1.0, protocol={5.5: 0.5})),
        (1.0, RadiotherapySpecification(alpha=1.0, protocol={0: 0.5})),
        (1.0, RadiotherapySpecification(alpha=1.0, protocol={10: 0.5})),
        (
            1.0,
            RadiotherapySpecification(alpha=1.0, protocol={t: 0.25 for t in range(10)}),
        ),
    ],
)
def test_radiotherapy_model(
    N_init: float, radiotherapy_specification: RadiotherapySpecification, plot=False
):

    theta = 1.0  # Maximum Cellularity
    dt = 1.0  # time step
    D = 0.0  # no diffusion
    k_0 = 0.0  # no proliferation

    domain_shape = (3, 3, 3)
    cellularity_array = N_init * np.ones(domain_shape)
    mask_array = np.ones(domain_shape)
    mask_array[0, :, :] = 0.0
    mask_array[-1, :, :] = 0
    mask_array[:, 0, :] = 0
    mask_array[:, -1, :] = 0
    mask_array[:, :, 0] = 0
    mask_array[:, :, -1] = 0

    k_array = k_0 * mask_array

    mask = NibabelNifti.from_array(mask_array)
    cellularity = NibabelNifti.from_array(cellularity_array)
    k = NibabelNifti.from_array(k_array)

    bcs = bound_condition_maker(mask)

    t_final = 10.0
    t_analytic, u_analytic = analytic_solution(
        N_init, list(np.arange(t_final + 1.0)), radiotherapy_specification
    )
    t_comp, results, _ = predict_cell_count(
        t=(0, t_final),
        dt=dt,
        initial_cellularity_image=cellularity,
        d=D,
        theta=theta,
        k=k,
        bcs=bcs,
        brain_mask=mask,
        radiotherapy_specification=radiotherapy_specification,
    )
    if plot:
        print([(t, u) for t, u in zip(t_analytic, u_analytic)])
        print([(t, r.array[1, 1, 1]) for t, r in zip(t_comp, results)])
        plt.plot(t_analytic, u_analytic, "k--", label="analytic")
        plt.plot(t_comp, [r.array[1, 1, 1] for r in results], label="computed")
        plt.legend()
        plt.show()

    assert len(t_analytic) == len(t_comp)
    assert mae(np.array(t_analytic), np.array(t_comp)) < 1e-8

    for t in range(len(t_analytic)):
        assert mae(results[t].array[1, 1, 1], u_analytic[t]) < 1e-2
