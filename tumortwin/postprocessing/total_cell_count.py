import numpy as np
import torch

from tumortwin.types.imaging import Image3D


def compute_total_cell_count(
    cellularity: Image3D | np.ndarray | torch.Tensor, carrying_capacity: float
) -> float | torch.Tensor:
    """
    Computes the total cell count in a given cellularity map.

    The total cell count is calculated as the sum of the cellularity values across
    the map, scaled by the voxel carrying capacity.

    Args:
        cellularity (Image3D | np.ndarray | torch.Tensor): The cellularity map,
            which can be an `Image3D` instance, a Numpy array, or a PyTorch tensor.
        carrying_capacity (float): The carrying capacity of a voxel.

    Returns:
        float | torch.Tensor: The total cell count. If the input cellularity is
            a PyTorch tensor, the result will also be a tensor.

    Raises:
        ValueError: If the input cellularity is of an unsupported type.
    """
    if isinstance(cellularity, Image3D):
        return carrying_capacity * np.sum(cellularity.array)
    elif isinstance(cellularity, np.ndarray):
        return carrying_capacity * np.sum(cellularity)
    elif isinstance(cellularity, torch.Tensor):
        return carrying_capacity * torch.sum(cellularity)
    else:
        raise ValueError("Unsupported type for input cellularity field!")
