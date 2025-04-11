import numpy as np

from tumortwin.preprocessing import ADC_to_cellularity
from tumortwin.types.imaging import Image3D


def test_cellularity_estimation(
    ADC: Image3D,
    roi_enhance: Image3D,
    roi_nonenhance: Image3D,
    matlab_parameters: np.ndarray,
) -> None:

    N_est = ADC_to_cellularity(ADC, roi_enhance, roi_nonenhance)
    N_est_original = matlab_parameters["N_est"]
    matches = 0
    for i in range(N_est.shape.z):
        if N_est.array[:, :, i].shape == N_est_original[:, :, i].shape and np.allclose(
            N_est.array[:, :, i], N_est_original[:, :, i]
        ):
            matches += 1
    assert np.sum(matches) == N_est.shape.z
