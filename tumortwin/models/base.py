import torch
import torch.nn as nn


class TumorGrowthModel3D(nn.Module):
    """
    A base class for 3D tumor growth models using PyTorch.

    This class defines the interface for tumor growth models that can simulate
    tumor dynamics in a 3D environment. Subclasses must implement the `forward`
    method to define the model-specific behavior.

    """

    def __init__(self):
        """
        Initializes the base class.

        This constructor calls the parent PyTorch `nn.Module` initializer.
        """
        super().__init__()

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the tumor growth model.

        Subclasses must implement this method to specify the model's behavior
        during the forward pass.

        Args:
            t (torch.Tensor): A tensor representing time points, typically of shape `(batch_size, 1)`.
            u (torch.Tensor): A tensor representing the input state, such as tumor properties
                or environmental variables, typically of shape `(batch_size, ...)`.

        Returns:
            du_dt (torch.Tensor): A tensor representing the computed output state, typically of the same shape as `u`.

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError
