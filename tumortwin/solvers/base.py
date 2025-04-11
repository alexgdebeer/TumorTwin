from datetime import datetime
from typing import List, Tuple

import torch


class ForwardSolver:
    """
    Abstract base class for forward solvers in tumor growth modeling.

    A forward solver numerically integrates the evolution of the tumor density field
    over specified timepoints, starting from an initial condition.

    Methods:
        solve(timepoints, u_initial):
            Computes the tumor density at each timepoint by solving the growth model.
            Must be implemented by subclasses.

    Raises:
        NotImplementedError: If a subclass does not implement the `solve` method.
    """

    def solve(
        self, timepoints: List[datetime], u_initial: torch.Tensor
    ) -> Tuple[List[datetime], List[torch.Tensor]]:
        """
        Abstract method to solve the tumor growth model.

        Args:
            timepoints (List[datetime]): List of timepoints at which the solution is desired.
            u_initial (torch.Tensor): Initial tumor density field.

        Returns:
            Tuple[List[datetime], List[torch.Tensor]]:
                - List of datetime objects corresponding to the solution timepoints.
                - List of torch.Tensor objects representing the tumor density at each timepoint.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError
