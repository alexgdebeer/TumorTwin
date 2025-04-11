import unittest

import numpy as np

from tumortwin.preprocessing.crop import (
    crop_array_to_bounding_box,
    crop_image_to_bounding_box,
    cropped_array_to_full,
    get_bounding_box,
    restrict_bounding_box,
)
from tumortwin.types.imaging import NibabelNifti


class TestCrop(unittest.TestCase):
    def test_crop_array(self):
        t1_nii = "data/t1_post_c.nii.gz"
        roi_nii = "data/roi_enhance.nii.gz"

        t1_img = NibabelNifti.from_file(t1_nii)
        roi_img = NibabelNifti.from_file(roi_nii)

        roi_bbox = get_bounding_box(roi_img.array, padding=20)
        t1_roi = crop_array_to_bounding_box(t1_img.array, roi_bbox)

        tmp = [roi_bbox[i].stop - roi_bbox[i].start for i in range(3)]
        roi_bbox_elems = np.prod(tmp)

        assert (
            np.size(t1_roi) == roi_bbox_elems
        ), "The cropped ROI image should have the same number of elements as the bounding box"

    def test_crop_img(self):
        t1_nii = "data/t1_post_c.nii.gz"
        brain_nii = "data/brainmask.nii.gz"

        brain_img = NibabelNifti.from_file(brain_nii)
        t1_img = NibabelNifti.from_file(t1_nii)

        brain_bbox = get_bounding_box(brain_img.array)
        t1_brain = crop_image_to_bounding_box(t1_img, brain_bbox)

        tmp = [brain_bbox[i].stop - brain_bbox[i].start for i in range(3)]
        brain_bbox_elems = np.prod(tmp)

        assert (
            np.size(t1_brain.array) == brain_bbox_elems
        ), "The cropped ROI image should have the same number of elements as the bounding box"

    def test_restrict_bbox(self):
        roi_nii = "data/roi_enhance.nii.gz"
        brain_nii = "data/brainmask.nii.gz"

        brain_img = NibabelNifti.from_file(brain_nii)
        roi_img = NibabelNifti.from_file(roi_nii)

        # get bounding boxes
        roi_bbox = get_bounding_box(roi_img.array, padding=20)
        brain_bbox = get_bounding_box(brain_img.array)

        # combine the two
        restricted_bbox = restrict_bounding_box(roi_bbox, brain_bbox)

        # the z-dimension should be set by the brain bounding box
        assert (
            restricted_bbox[-1] == brain_bbox[-1]
        ), "The brain bounding box should restrict the z-dimension of the ROI bounding box"

    def test_crop_to_full(self):
        t1_nii = "data/t1_post_c.nii.gz"
        brain_nii = "data/brainmask.nii.gz"

        t1_img = NibabelNifti.from_file(t1_nii)
        brain_img = NibabelNifti.from_file(brain_nii)

        brain_bbox = get_bounding_box(brain_img.array)

        # develop reference image
        t1_ref = 0.0 * t1_img.array
        t1_ref[brain_bbox] = t1_img.array[brain_bbox]

        # crop the image by bounding box
        brain_bbox = get_bounding_box(brain_img.array)
        t1_brain = crop_image_to_bounding_box(t1_img, brain_bbox)
        t1_brain_to_full = cropped_array_to_full(
            t1_brain.array, t1_img.array.shape, brain_bbox
        )

        assert np.array_equal(
            t1_ref, t1_brain_to_full
        ), "The rasterized cropped image should be the same as the reference image"
