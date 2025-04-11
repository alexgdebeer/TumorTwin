import unittest

import numpy as np

from tumortwin.qoi.qoi import (
    compute_ccc,
    computeVoxelCCC,
    computeVoxelDice,
    computeVoxelTTC,
    computeVoxelTTV,
)
from tumortwin.types.imaging import NibabelNifti


class TestCCC(unittest.TestCase):
    def test_ccc(self):
        # test the compute_ccc function
        # Test case is from: https://rowannicholls.github.io/python/statistics/agreement/concordance_correlation_coefficient.html
        # which uses: https://github.com/stylianos-kampakis/supervisedPCA-Python/blob/master/Untitled.py

        x = np.array([2.5, 0.0, 2, 8])
        y = np.array([3, -0.5, 2, 7])

        ccc = compute_ccc(x, y, bias=False, use_pearson=True)

        VAL = 0.97678916827853024  # expected value

        assert np.isclose(ccc, VAL, atol=1e-10)

    def test_voxelCCC(self):
        adc_nii = "data/adc.nii.gz"
        roi_nii = "data/roi_enhance.nii.gz"

        adc_img = NibabelNifti.from_file(adc_nii)
        roi_img = NibabelNifti.from_file(roi_nii)

        ccc = computeVoxelCCC(adc_img, adc_img, roi_img)

        # CCC should be 1 when comparing the same image.
        assert np.isclose(ccc, 1.0, atol=1e-10)

    def test_dice(self):
        roi_nii = "data/roi_enhance.nii.gz"

        roi_img = NibabelNifti.from_file(roi_nii)

        dice = computeVoxelDice(roi_img, roi_img)

        # Dice coefficient should be 1 when comparing the same image.
        assert np.isclose(dice, 1.0, atol=1e-10)

    def test_ttv(self):
        # Test the computeVoxelCCC function with different images
        roi_nii = "data/roi_enhance.nii.gz"
        roi_img = NibabelNifti.from_file(roi_nii)

        ttv = computeVoxelTTV(roi_img, threshold=0.5)

        VAL = 11443.5  # expected value with 0.5 threshold (shouldn't matter with the ROI image)
        assert np.isclose(ttv, VAL, atol=1e-10)

    def test_ttc(self):
        roi_nii = "data/roi_enhance.nii.gz"
        roi_img = NibabelNifti.from_file(roi_nii)

        carrycap = 10.0  # dummy carrying capacity
        ttc = computeVoxelTTC(roi_img, carrycap=carrycap)

        # to compare against the expected value from the ttv test
        VAL = 10.0 * 11443.5 / roi_img.voxvol  # expected value

        assert np.isclose(ttc, VAL, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
