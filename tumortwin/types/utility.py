from enum import Enum
from typing import Tuple, TypeAlias


class Boundary(Enum):
    """
    Enum for specifying boundary conditions.

    Attributes:
        NONE (int): No boundary condition (outside of brain).
        BACKWARD (int): Backward boundary condition (point n+1 is outside, while n, n-1 inside).
        INTERIOR (int): Interior boundary condition (interior of brain).
        FORWARD (int): Forward boundary condition (point n-1 is outside, while n, n+1 inside).
    """

    NONE = 0
    BACKWARD = 1
    INTERIOR = 2
    FORWARD = 3


class Direction(Enum):
    """
    Enumeration for specifying spatial directions.

    Attributes:
        X (int): X-axis direction.
        Y (int): Y-axis direction.
        Z (int): Z-axis direction.
    """

    X = 0
    Y = 1
    Z = 2


# List of all possible directions
DIRECTIONS = [Direction.X, Direction.Y, Direction.Z]

# Type alias for a 3D index, which can include slices or integers for each axis
Index3d: TypeAlias = Tuple[slice | int, slice | int, slice | int]
