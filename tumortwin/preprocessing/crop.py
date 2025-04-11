import itertools
from typing import Tuple, TypeAlias, TypeVar

import numpy as np

from tumortwin.types.imaging import Image3D

BoundingBoxIndices: TypeAlias = Tuple
# We are using numpy slice types, so that you can do array[indices] where array is an N-D array, and indices is a BoundingBoxIndices type, which is a Tuple of length N.
# Unfortunately, there isn't a more explicit type hint for the N-D slice object.


def get_bounding_box(mask_array: np.ndarray, padding: int = 1) -> BoundingBoxIndices:
    """Get the bounding box of a binary mask array.

    Args:
        mask_array (np.ndarray): A binary mask array.
        pad (int, optional): How many voxels to pad the bounding box by. Defaults to 1.

    Returns:
        BoundingBoxIndices: Tuple of slice objects defining the bounding box.
    """
    # returns a numpy slice expression, i.e. a tuple of mask_array.dim slices
    N = mask_array.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(mask_array, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])

    return np.s_[
        max(out[0] - padding, 0) : min(mask_array.shape[0], out[1] + padding + 1),
        max(out[2] - padding, 0) : min(mask_array.shape[1], out[3] + padding + 1),
        max(out[4] - padding, 0) : min(mask_array.shape[2], out[5] + padding + 1),
    ]


def crop_array_to_bounding_box(
    array: np.ndarray, bounding_box: BoundingBoxIndices
) -> np.ndarray:
    """
    Crop an input array to the specified bounding box.

    Args:
        array (np.ndarray): The input array to crop.
        bounding_box (BoundingBoxIndices): The bounding box, defined as slices
            for each dimension of the array.

    Returns:
        np.ndarray: The cropped array.

    Raises:
        ValueError: If the bounding box dimensions are incompatible with the
            input array.
    """
    try:
        return array[bounding_box]
    except:
        ValueError(
            f"Bounding box and input array are incompatible, likely different dimensions! Array dimension: {array.ndim}, BoundingBox dimension: {len(bounding_box)}"
        )


def restrict_bounding_box(
    small_bbox: BoundingBoxIndices, large_bbox: BoundingBoxIndices
) -> BoundingBoxIndices:
    """This restricts a bounding box by the domain of another bounding box.
    For example, when restricting an ROI bounding box to the domain of a brain bounding box.

    Args:
        small_bbox (BoundingBoxIndices): The smaller bounding box.
        large_bbox (BoundingBoxIndices): The larger bounding box.

    Returns:
        BoundingBoxIndices: The restricted bounding box.
    """
    return tuple(
        slice(
            max(small_bbox[i].start, large_bbox[i].start),
            min(small_bbox[i].stop, large_bbox[i].stop),
            small_bbox[i].step,
        )
        for i in range(len(small_bbox))
    )


T = TypeVar("T", bound=Image3D)


def crop_image_to_bounding_box(image: T, bounding_box: BoundingBoxIndices) -> T:
    """
    Crop an Image3D object to a specified bounding box.

    This function modifies the `Image3D` instance by cropping its internal
    array data to the provided bounding box and returns a new instance.

    Args:
        image (Image3D): The input `Image3D` object.
        bounding_box (BoundingBoxIndices): The bounding box to crop to.

    Returns:
        Image3D: A new `Image3D` instance cropped to the specified bounding box.
    """
    return image.__class__.from_array(
        crop_array_to_bounding_box(image.array, bounding_box), image
    )


def cropped_array_to_full(
    cropped_array: np.ndarray,
    full_shape: Tuple[int, int, int],
    bounding_box: BoundingBoxIndices,
) -> np.ndarray:
    full_array = np.zeros(full_shape, dtype=cropped_array.dtype)
    full_array[bounding_box] = cropped_array
    return full_array
