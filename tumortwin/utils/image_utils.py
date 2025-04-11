import numpy as np


def find_best_slice(input_array: np.ndarray) -> int:
    """
    Finds the best z-index slice based on the sum of voxel intensities across the x and y axes.

    Args:
        input_array (np.ndarray): A 3D numpy array representing the image.

    Returns:
        int: The z-index of the slice with the highest summed intensity.
    """
    zVols = np.sum(input_array, axis=(0, 1))
    return int(np.argmax(zVols))
