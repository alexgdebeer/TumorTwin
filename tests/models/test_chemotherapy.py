from typing import List

import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from tumortwin.preprocessing.bound_condition_maker import bound_condition_maker
from tumortwin.solvers.predict_cell_count import predict_cell_count
from tumortwin.treatments.chemotherapy import compute_total_cell_death_chemo
from tumortwin.types.imaging import NibabelNifti
from tumortwin.types.treatment import ChemotherapySpecification


def analytic_solution(t_span, U_0, chemotherapy_specifications):
    def rhs(t, _):
        return -1.0 * compute_total_cell_death_chemo(t, chemotherapy_specifications)

    sol = solve_ivp(
        rhs, (t_span[0], t_span[-1]), [U_0], t_eval=t_span, method="RK45", rtol=1e-6
    )
    return sol.t, sol.y[0]


def mae(u, v):
    return np.mean(np.abs(np.array(u) - np.array(v)))


@pytest.mark.parametrize(
    "U_0, chemotherapy_specifications",
    [
        (
            0.5,
            [
                ChemotherapySpecification(
                    sensitivity=0.01, decay_rate=0.0, protocol={0.0: 1.0}
                )
            ],
        ),
        (
            0.5,
            [
                ChemotherapySpecification(
                    sensitivity=0.2, decay_rate=0.9, protocol={0.0: 1.0}
                )
            ],
        ),
        (
            0.5,
            [
                ChemotherapySpecification(
                    sensitivity=0.2, decay_rate=0.9, protocol={2.5: 1.0}
                )
            ],
        ),
        (
            0.5,
            [
                ChemotherapySpecification(
                    sensitivity=0.2, decay_rate=0.9, protocol={0.0: 0.1}
                ),
                ChemotherapySpecification(
                    sensitivity=0.2, decay_rate=0.9, protocol={2.5: 1.0}
                ),
            ],
        ),
    ],
)
def test_chemotherapy_model(
    U_0: float, chemotherapy_specifications: List[ChemotherapySpecification], plot=False
):
    dt = 0.01
    ts = np.arange(0, 5 + dt, dt)

    theta = 1.0  # Maximum Cellularity
    k_0 = 0.0
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

    _, u_truth = analytic_solution(ts, U_0, chemotherapy_specifications)

    _, results, _ = predict_cell_count(
        t=(ts[0], ts[-1]),
        dt=dt,
        initial_cellularity_image=cellularity,
        d=D,
        theta=theta,
        k=k,
        bcs=bcs,
        brain_mask=mask,
        chemotherapy_specifications=chemotherapy_specifications,
    )
    u_computed = [r.array[1, 1, 1] for r in results]

    if plot:
        plt.plot(ts, u_truth, "k--", label="truth")
        plt.plot(ts, u_computed, label="computed")

    for t in ts:
        assert mae(u_computed, u_truth) < 1e-2
