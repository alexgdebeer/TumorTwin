import numpy as np

from tumortwin.types.imaging import Image3D
from tumortwin.types.utility import Boundary, Direction


def bound_condition_maker(mask: Image3D) -> Image3D:
    """
    Assigns boundary conditions to a binary mask and returns a 4D array representing
    the boundary condition at each voxel in three orthogonal directions.

    This function processes a 3D binary mask, calculating boundary conditions for
    each voxel. Boundary conditions can be `INTERIOR`, `FORWARD`, `BACKWARD`, or
    `NONE`, depending on the presence of neighboring voxels.

    Args:
        mask (Image3D): A binary mask where non-zero values represent the region
            of interest.

    Returns:
        Image3D: A 4D array (same spatial dimensions as `mask`, with an additional
            axis for the X, Y, and Z directions) containing boundary condition
            values for each voxel and direction.
    """
    return mask.__class__.from_array(
        bound_condition_maker_inner(
            mask.array,
            np.pad(mask.array, pad_width=1, mode="constant", constant_values=0),
        ),
        mask,
    )


def check_neighbors(
    mask: np.ndarray, x: int, y: int, z: int, direction: Direction
) -> int:
    """
    Checks the neighbors of a voxel in a specific direction and determines its boundary condition.

    Args:
        mask (np.ndarray): A binary mask (3D array).
        x (int): X-coordinate of the voxel.
        y (int): Y-coordinate of the voxel.
        z (int): Z-coordinate of the voxel.
        direction (Direction): The direction to check (`X`, `Y`, or `Z`).

    Returns:
        int: The boundary condition value (e.g., `Boundary.INTERIOR.value`, `Boundary.FORWARD.value`,
            `Boundary.BACKWARD.value`, or `Boundary.NONE.value`).
    """
    match direction:
        case Direction.X:
            if mask[x - 1, y, z] and mask[x + 1, y, z]:
                return Boundary.INTERIOR.value
            elif mask[x - 1, y, z] and mask[x - 2, y, z]:
                return Boundary.BACKWARD.value
            elif mask[x + 1, y, z] and mask[x + 2, y, z]:
                return Boundary.FORWARD.value
            else:
                return Boundary.NONE.value
        case Direction.Y:
            if mask[x, y - 1, z] and mask[x, y + 1, z]:
                return Boundary.INTERIOR.value
            elif mask[x, y - 1, z] and mask[x, y - 2, z]:
                return Boundary.BACKWARD.value
            elif mask[x, y + 1, z] and mask[x, y + 2, z]:
                return Boundary.FORWARD.value
            else:
                return Boundary.NONE.value
        case Direction.Z:
            if mask[x, y, z - 1] and mask[x, y, z + 1]:
                return Boundary.INTERIOR.value
            elif mask[x, y, z - 1] and mask[x, y, z - 2]:
                return Boundary.BACKWARD.value
            elif mask[x, y, z + 1] and mask[x, y, z + 2]:
                return Boundary.FORWARD.value
            else:
                return Boundary.NONE.value


def bound_condition_maker_inner(
    mask: np.ndarray, padded_mask: np.ndarray
) -> np.ndarray:
    """
    Core implementation for computing boundary conditions.

    This function iterates over all non-zero voxels in the mask and determines the
    boundary condition for each voxel in the X, Y, and Z directions.

    Args:
        mask (np.ndarray): A 3D binary mask.
        padded_mask (np.ndarray): The same mask padded by one voxel in each
            dimension, to simplify boundary condition calculations.

    Returns:
        np.ndarray: A 4D array of the same spatial dimensions as `mask`, with an
            additional axis for boundary conditions in three directions.
    """
    sx, sy, sz = mask.shape

    # Initialize boundary condition array with `Boundary.NONE.value`
    bcs = np.ones((sx, sy, sz, 3)) * Boundary.NONE.value

    # Iterate over non-zero voxels in the mask
    for x, y, z in np.argwhere(mask):
        if mask[x, y, z]:
            # Compute boundary conditions for each direction
            bcs[x, y, z, 0] = check_neighbors(
                padded_mask, x + 1, y + 1, z + 1, Direction.X
            )
            bcs[x, y, z, 1] = check_neighbors(
                padded_mask, x + 1, y + 1, z + 1, Direction.Y
            )
            bcs[x, y, z, 2] = check_neighbors(
                padded_mask, x + 1, y + 1, z + 1, Direction.Z
            )
    return bcs
