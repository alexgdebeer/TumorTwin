"""The imaging module contains types for working with medical imaging data."""

from __future__ import annotations

from enum import Enum
from typing import Any, NamedTuple, Optional, TypeVar

import nibabel as nib
import numpy as np
import torch
from nibabel.nifti1 import Nifti1Image
from pydantic import BaseModel, ConfigDict, Field, FilePath


class Unit(Enum):
    """
    Enum representing units of measurement commonly used in medical imaging.

    Attributes:
        UNKNOWN (int): Unknown unit.
        METER (int): Unit of length in meters.
        MILLIMETER (int): Unit of length in millimeters.
        MICRON (int): Unit of length in microns.
        SECONDS (int): Unit of time in seconds.
        MILLISECONDS (int): Unit of time in milliseconds.
        MICROSECONDS (int): Unit of time in microseconds.
        HERTZ (int): Unit of frequency in hertz.
        PARTSPERMILLION (int): Unit for parts per million.
        RADIANSPERSECOND (int): Unit for angular velocity in radians per second.
    """

    UNKNOWN = 0
    METER = 1
    MILLIMETER = 2
    MICRON = 3
    SECONDS = 8
    MILLISECONDS = 16
    MICROSECONDS = 24
    HERTZ = 32
    PARTSPERMILLION = 40
    RADIANSPERSECOND = 48


class Shape3D(NamedTuple):
    """
    Represents the 3D size/shape of an image or volume.

    Attributes:
        x (int): The size (i.e. number of points) along the x-axis.
        y (int): The size (i.e. number of points) along the y-axis.
        z (int): The size (i.e. number of points) along the z-axis.
    """

    x: int
    y: int
    z: int


class Spacing3D(NamedTuple):
    """
    Represents the spacing between voxels in a 3D image or volume.

    Attributes:
        x (float): Spacing along the x-axis.
        y (float): Spacing along the y-axis.
        z (float): Spacing along the z-axis.
        unit (Unit): The unit of measurement for the spacing values.
    """

    x: float
    y: float
    z: float
    unit: Unit


class Image3D(BaseModel):
    """
    Abstract base class for 3D medical images.

    Attributes:
        image (Any): The underlying image data (e.g., NIfTI image object).

    Methods:
        array: Abstract property to get the image data as a NumPy array or PyTorch tensor.
        from_file: Abstract method to load an image from a file.
        to_file: Abstract method to save the image to a file.
        shape: Abstract property to get the shape of the image.
        centroid: Abstract property to get the centroid of the image.
        spacing: Abstract property to get the voxel spacing of the image.
        voxvol: Abstract property to calculate the volume of a voxel.
        from_array: Abstract method to create an image from a NumPy array or PyTorch tensor.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Any

    @property
    def array(self) -> np.ndarray | torch.Tensor:
        """
        Returns the image data as a NumPy array or PyTorch tensor.

        Returns:
            np.ndarray | torch.Tensor: The image data.
        """
        raise NotImplementedError

    @staticmethod
    def from_file(path: FilePath) -> Image3D:
        """
        Loads a 3D image from a file.

        Args:
            path (FilePath): The path to the image file.

        Returns:
            Image3D: The loaded image.
        """
        raise NotImplementedError

    def to_file(self, path: FilePath) -> None:
        """
        Saves the image to a file.

        Args:
            path (FilePath): The destination file path.
        """
        raise NotImplementedError

    @property
    def shape(self) -> Shape3D:
        """
        Returns the shape of the image.

        Returns:
            Shape3D: The 3D shape of the image.
        """
        raise NotImplementedError

    @property
    def centroid(self) -> np.ndarray:
        """
        Returns the centroid of the image.

        Returns:
            np.ndarray: The centroid coordinates.
        """
        raise NotImplementedError

    @property
    def spacing(self) -> Spacing3D:
        """
        Returns the voxel spacing of the image.

        Returns:
            Spacing3D: The spacing along each axis and the unit of measurement.
        """
        raise NotImplementedError

    @property
    def voxvol(self) -> float:
        """
        Calculates the volume of a voxel.

        Returns:
            float: The voxel volume.
        """
        return self.spacing.x * self.spacing.y * self.spacing.z

    @staticmethod
    def from_array(
        array: np.ndarray | torch.Tensor, referenceImage: Image3D
    ) -> Image3D:
        """
        Creates a 3D image from a NumPy array or PyTorch tensor.

        Args:
            array (np.ndarray | torch.Tensor): The array representing the image.
            referenceImage (Image3D): An reference image for copying metadata.

        Returns:
            Image3D: The created 3D image.
        """
        raise NotImplementedError


class NibabelNifti(Image3D):
    """
    Implementation of Image3D for NIfTI images using NiBabel.

    Attributes:
        image (Nifti1Image): The NIfTI image object.
    """

    image: Nifti1Image

    @property
    def array(self) -> np.ndarray | torch.Tensor:
        """
        Returns the image data as a NumPy array.

        Returns:
            np.ndarray | torch.Tensor: The image data.
        """
        return self.image.get_fdata()

    @property
    def shape(self) -> Shape3D:
        """
        Returns the shape of the image.

        Returns:
            Shape3D: The 3D shape of the image.
        """
        return Shape3D(self.image.shape[0], self.image.shape[1], self.image.shape[2])

    @property
    def spacing(self) -> Spacing3D:
        """
        Returns the voxel spacing of the image.

        Returns:
            Spacing3D: The spacing along each axis and the unit of measurement.
        """
        return Spacing3D(
            self.image.header["pixdim"][1],
            y=self.image.header["pixdim"][2],
            z=self.image.header["pixdim"][3],
            unit=Unit(self.image.header["xyzt_units"] % 8),
        )

    @property
    def centroid(self) -> np.ndarray:
        """
        Returns the centroid of the image.

        Returns:
            np.ndarray: The centroid coordinates.
        """
        return self.image.affine[:, -1][:-1]  # get the centroid from the affine matrix

    @staticmethod
    def from_file(path: FilePath) -> NibabelNifti:
        """
        Loads a NIfTI image from a file.

        Args:
            path (FilePath): The path to the NIfTI file.

        Returns:
            NibabelNifti: The loaded NIfTI image.
        """
        return NibabelNifti(image=nib.nifti1.load(path))

    def to_file(self, path: FilePath) -> None:
        """
        Saves the NIfTI image to a file.

        Args:
            path (FilePath): The destination file path.
        """
        return nib.nifti1.save(self.image, path)

    @staticmethod
    def from_array(
        array: np.ndarray | torch.Tensor, referenceImage: Optional[NibabelNifti] = None
    ) -> NibabelNifti:
        """
        Creates a NIfTI image from a NumPy array or PyTorch tensor.

        Args:
            array (np.ndarray | torch.Tensor): The array representing the image.
            referenceImage (Optional[NibabelNifti]): An reference NIfTI image for copying metadata.

        Returns:
            NibabelNifti: The created NIfTI image.
        """
        arr = array.detach().numpy() if isinstance(array, torch.Tensor) else array
        if referenceImage is None:
            return NibabelNifti(image=Nifti1Image(arr, affine=np.eye(4)))
        else:
            return NibabelNifti(
                image=Nifti1Image(
                    arr,
                    affine=referenceImage.image.affine,
                    header=referenceImage.image.header,
                )
            )


class ITKNifti(Image3D):
    """
    Placeholder class for ITK-based NIfTI handling.
    """

    pass


AnyImage3D = TypeVar("AnyImage3D", bound=Image3D)


class CropTarget(Enum):
    """
    Enum representing different cropping targets for medical imaging.

    Attributes:
        ANATOMY (int): Crop to the anatomical structure (e.g., brain or breast)
        ROI_ENHANCE (int): Crop to the region of interest (ROI) for the enhancing region.
        ROI_NONENHANCE (int): Crop to the region of interest (ROI) for the non-enhancing region.
    """

    ANATOMY = 1
    ROI_ENHANCE = 2
    ROI_NONENHANCE = 3


class CropSettings(BaseModel):
    """
    Configuration settings for cropping medical images.

    Attributes:
        crop_to (CropTarget): The target area to crop. Defaults to `CropTarget.ANATOMY`.
        padding (int): The amount of padding to apply around the cropped area. Defaults to 1.
        visit_index (Optional[int]): Specifies which visit's ROI to use when `crop_to` is
            `ROI_ENHANCE` or `ROI_NONENHANCE`. If `None`, the union of all available ROIs is used.
            If an integer is provided, only the ROI from that specific visit is used.
    """

    crop_to: CropTarget = Field(default=CropTarget.ANATOMY)
    padding: int = Field(default=1)
    visit_index: Optional[int] = Field(default=None)
