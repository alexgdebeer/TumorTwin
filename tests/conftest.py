import os
from pathlib import Path

import numpy as np
import pytest
import scipy as sp

from tumortwin.types.imaging import NibabelNifti


@pytest.fixture()
def matlab_parameters(request) -> np.ndarray:
    test_dir = os.path.dirname(request.module.__file__)
    data_filename = os.path.join(test_dir, "data", "matlab_parameters.mat")
    return sp.io.loadmat(data_filename)


@pytest.fixture()
def brain_mask(request) -> NibabelNifti:
    test_dir = os.path.dirname(request.module.__file__)
    data_filename = os.path.join(test_dir, "data", "brainmask.nii.gz")
    return NibabelNifti.from_file(Path(data_filename))


@pytest.fixture()
def ADC(request) -> NibabelNifti:
    test_dir = os.path.dirname(request.module.__file__)
    data_filename = os.path.join(test_dir, "data", "adc.nii.gz")
    return NibabelNifti.from_file(Path(data_filename))


@pytest.fixture()
def roi_enhance(request) -> NibabelNifti:
    test_dir = os.path.dirname(request.module.__file__)
    data_filename = os.path.join(test_dir, "data", "roi_enhance.nii.gz")
    return NibabelNifti.from_file(Path(data_filename))


@pytest.fixture()
def roi_nonenhance(request) -> NibabelNifti:
    test_dir = os.path.dirname(request.module.__file__)
    data_filename = os.path.join(test_dir, "data", "roi_nonenhance.nii.gz")
    return NibabelNifti.from_file(Path(data_filename))
