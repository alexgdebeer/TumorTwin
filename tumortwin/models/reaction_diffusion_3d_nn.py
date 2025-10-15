from datetime import datetime, timedelta
from typing import List, Optional, Union

import torch
import torch.nn as nn
import tqdm.auto as tqdm

from tumortwin.models.base import TumorGrowthModel3D
from tumortwin.preprocessing import bound_condition_maker
from tumortwin.treatments import (
    compute_radiotherapy_cell_survival_fraction,
    compute_total_cell_death_chemo,
)
from tumortwin.types import (
    ChemotherapySpecification,
    HGGPatientData,
    RadiotherapySpecification,
    TNBCPatientData,
)
from tumortwin.types.utility import Boundary
    
class ReactionDiffusion3D(TumorGrowthModel3D):
    """
    A 3D reaction-diffusion model for simulating tumor growth.

    This class extends the `TumorGrowthModel3D` base class to implement a reaction-diffusion
    framework with support for radiotherapy and chemotherapy. The model incorporates spatial
    diffusion, proliferation dynamics, and treatment effects.

    Attributes:
        k (torch.Tensor): Tumor growth rate parameter (proliferation rate).
        d (torch.Tensor): Tumor diffusion coefficient (spatial spread rate).
        theta (torch.Tensor): Carrying capacity of the tumor cells (maximum density).
        bcs (torch.Tensor): Boundary conditions derived from the patient brain mask.
        brain_mask (torch.Tensor): Binary mask indicating the brain region.
        radiotherapy_specification (Optional[RadiotherapySpecification]): Specification of radiotherapy protocol.
        radiotherapy_days (Optional[dict]): Dictionary mapping days (since initial time) to radiotherapy doses.
        chemotherapy_specifications (Optional[List[ChemotherapySpecification]]): List of chemotherapy protocols.
        t_initial (datetime): Initial time of the simulation.
        fd_stencil_backward_coeff (List[torch.Tensor]): Backward finite-difference coefficients for each axis.
        fd_stencil_central_coeff (List[torch.Tensor]): Central finite-difference coefficients for each axis.
        fd_stencil_forward_coeff (List[torch.Tensor]): Forward finite-difference coefficients for each axis.
        device (torch.device): The device on which to perform computations (e.g., CPU or GPU).

    Methods:
        __init__: Initializes the model with parameters and treatment specifications.
        forward: Computes the rate of change of tumor density at a given time step.
        callback_step: Applies treatment effects (e.g., radiotherapy) and ensures constraints on the tumor density.
        _compute_laplacian: Computes the spatial Laplacian of the tumor density field.
        _backward_slice: Extracts the backward slice along a given axis.
        _central_slice: Extracts the central slice along a given axis.
        _forward_slice: Extracts the forward slice along a given axis.
    """

    def __init__(
        self,
        k: Union[nn.Module, torch.Tensor],
        d: torch.Tensor,
        theta: torch.Tensor,
        patient_data: Union[HGGPatientData, TNBCPatientData],
        initial_time: datetime,
        *,
        radiotherapy_specification: Optional[RadiotherapySpecification] = None,
        chemotherapy_specifications: Optional[List[ChemotherapySpecification]] = None,
        radiotherapy_days=None,
        require_grad: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initializes the ReactionDiffusion3D model with patient-specific data and parameters.

        Args:
            k (torch.Tensor): Tumor proliferation rate tensor.
            d (torch.Tensor): Tumor diffusion coefficient tensor.
            theta (torch.Tensor): Tumor carrying capacity tensor.
            patient_data (HGGPatientData): Patient-specific data including brain mask and imaging.
            initial_time (datetime): Initial simulation time.
            radiotherapy_specification (Optional[RadiotherapySpecification]): Radiotherapy protocol.
            chemotherapy_specifications (Optional[List[ChemotherapySpecification]]): Chemotherapy protocols.
            radiotherapy_days (Optional[dict]): Mapping of radiotherapy days to doses.
            require_grad (bool): Whether the parameters `k` and `d` require gradients.
        """
        super().__init__()
        self.device = device
        # self.k = nn.Parameter(k.to(device), requires_grad=require_grad)
        self.k = k.requires_grad_(require_grad)
        self.d = nn.Parameter(d.to(device), requires_grad=require_grad)
        self.theta = theta.to(device)
        mask_image = (
            patient_data.breastmask_image
            if hasattr(patient_data, "breastmask_image")
            else patient_data.brainmask_image
        )
        self.bcs = torch.from_numpy(bound_condition_maker(mask_image).array)
        self.comp_mask = torch.from_numpy(mask_image.array)
        self.radiotherapy_specification = radiotherapy_specification
        if radiotherapy_specification:
            self.radiotherapy_days = dict(
                [
                    (float((day - initial_time).days), dose)
                    for day, dose in radiotherapy_specification.protocol.items()
                ]
            )
        self.chemotherapy_specifications = chemotherapy_specifications
        self.ct_sens = nn.ParameterList(
            [spec.sensitivity for spec in self.chemotherapy_specifications or []]
        )
        self.t_initial = initial_time

        # Compute finite-difference stencils based on image spacing and boundary conditions
        spacing = mask_image.spacing

        # Create 1D tensors for each dimension
        x_coords = torch.arange(0, mask_image.shape[0]*spacing.x, spacing.x)
        y_coords = torch.arange(0, mask_image.shape[1]*spacing.y, spacing.y)
        z_coords = torch.arange(0, mask_image.shape[2]*spacing.z, spacing.z)

        # Use meshgrid to create all combinations of coordinates
        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        self.voxel_coords = torch.stack([x_grid, y_grid, z_grid], dim=-1).to(device)
        self.voxel_coords = self.voxel_coords.reshape(-1, 3)

        image_spacing = [spacing.x, spacing.y, spacing.z]

        self.fd_stencil_backward_coeff = [[] for _ in range(len(image_spacing))]
        self.fd_stencil_central_coeff = [[] for _ in range(len(image_spacing))]
        self.fd_stencil_forward_coeff = [[] for _ in range(len(image_spacing))]
        for ax in [0, 1, 2]:
            back_mask = self.bcs[:, :, :, ax] == Boundary.BACKWARD.value
            interior_mask = self.bcs[:, :, :, ax] == Boundary.INTERIOR.value
            forward_mask = self.bcs[:, :, :, ax] == Boundary.FORWARD.value
            self.fd_stencil_backward_coeff[ax] = (
                self._central_slice(back_mask, ax)
                * 2.0
                / (image_spacing[ax] * image_spacing[ax])
            )
            self.fd_stencil_central_coeff[ax] = (
                self._central_slice(interior_mask, ax)
                * 1.0
                / (image_spacing[ax] * image_spacing[ax])
            )
            self.fd_stencil_forward_coeff[ax] = (
                self._central_slice(forward_mask, ax)
                * 2.0
                / (image_spacing[ax] * image_spacing[ax])
            )
        self.progress_bar: Optional[tqdm.tqdm] = None

    @torch.enable_grad()
    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Computes the rate of change of tumor density at a given time step.

        Args:
            t (torch.Tensor): Current simulation time (in days).
            u (torch.Tensor): Tumor density field.

        Returns:
            torch.Tensor: Rate of change of tumor density.
        """
        if self.t_initial is None:
            raise ValueError(
                "Unable to compute chemotherapy effect. No initial time set!"
            )
        chemotherapy_effect = (
            compute_total_cell_death_chemo(
                self.t_initial + timedelta(days=t.item()),
                self.chemotherapy_specifications,
            )
            if self.chemotherapy_specifications
            else None
        )
        device = self.device
        u = u.to(device)
        self.theta = self.theta.to(device)

        diffusion_term = self.d * self._compute_laplacian(u)
        if isinstance(self.k,torch.Tensor):
            proliferation_term = torch.multiply(
                u, torch.multiply(self.k, (1.0 - torch.clamp(u, 0.0, 1.0) / self.theta))
            )
        else:
            k_vox_values = self.k(self.voxel_coords).reshape(u.shape)
            proliferation_term = torch.multiply(
                u, torch.multiply(k_vox_values, (1.0 - torch.clamp(u, 0.0, 1.0) / self.theta))
            )
        chemotherapy_term = chemotherapy_effect * u if chemotherapy_effect else 0.0
        dudt = diffusion_term + proliferation_term - chemotherapy_term
        return dudt

    def callback_step(self, t, u, dt):
        """
        Handles per-step updates during the simulation, including applying treatment effects.

        This method updates the tumor density field by applying radiotherapy effects (if specified)
        and ensures the tumor density is clamped within valid bounds.

        Args:
            t (torch.Tensor): Current simulation time.
            u (torch.Tensor): Tumor density field.
            dt (torch.Tensor): Time step duration.

        Returns:
            u (torch.Tensor): Updated tumor density field.
        """
        if self.progress_bar:
            self.progress_bar.update(dt.item())
        if (
            self.radiotherapy_specification is not None
            and float(t) in self.radiotherapy_days
        ):
            u *= compute_radiotherapy_cell_survival_fraction(
                self.radiotherapy_specification, self.radiotherapy_days[float(t)]
            )
        # Zero out tumor density outside the brain mask
        u[self.comp_mask == 0] = 0.0

        # Clamp tumor density within valid bounds [0, 1]
        torch.clamp_(u, 0.0, 1.0)
        return u

    def callback_step_adjoint(self, t, u, dt):
        """
        Handles per-step updates during the backward pass of the adjoint method.

        This method updates the adjoint variables by applying radiotherapy effects (if specified).

        Args:
            t (torch.Tensor): Current simulation time.
            u (torch.Tensor): Adjoint variables.
            dt (torch.Tensor): Time step duration.

        Returns:
            u (torch.Tensor): Updated adjoint variables.
        """
        if (
            self.radiotherapy_specification is not None
            and float(t) in self.radiotherapy_days
        ):

            RT_effect = compute_radiotherapy_cell_survival_fraction(
                self.radiotherapy_specification, self.radiotherapy_days[float(t)]
            )
            u_adj = u[2]
            u_adj *= RT_effect
        return u

    def _compute_laplacian(
        self,
        N: torch.Tensor,
    ):
        """
        Computes the spatial Laplacian of the tumor density field.

        The Laplacian is approximated using finite-difference stencils for each axis.

        Args:
            N (torch.Tensor): Tumor density field.

        Returns:
            torch.Tensor: Spatial Laplacian of the tumor density field.
        """

        laplacian = torch.zeros_like(N)

        for ax in [0, 1, 2]:
            backward_coeff = self.fd_stencil_backward_coeff[ax].to(N.device)
            central_coeff = self.fd_stencil_central_coeff[ax].to(N.device)
            forward_coeff = self.fd_stencil_forward_coeff[ax].to(N.device)

            a = self._central_slice(laplacian, ax)
            a += (
                backward_coeff
                * (self._backward_slice(N, ax) - self._central_slice(N, ax))
                + central_coeff
                * (
                    self._backward_slice(N, ax)
                    - 2.0 * self._central_slice(N, ax)
                    + self._forward_slice(N, ax)
                )
                + forward_coeff
                * (self._forward_slice(N, ax) - self._central_slice(N, ax))
            )
        return laplacian

    def _backward_slice(self, x: torch.Tensor, ax: int):
        """
        Extracts the backward slice along a specified axis.

        Args:
            x (torch.Tensor): Input tensor.
            ax (int): Axis along which to extract the backward slice.

        Returns:
            torch.Tensor: Backward slice.
        """
        return torch.narrow(x, ax, 0, x.shape[ax] - 2)

    def _central_slice(self, x: torch.Tensor, ax: int):
        """
        Extracts the central slice along a specified axis.

        Args:
            x (torch.Tensor): Input tensor.
            ax (int): Axis along which to extract the central slice.

        Returns:
            torch.Tensor: Central slice.
        """
        return torch.narrow(x, ax, 1, x.shape[ax] - 2)

    def _forward_slice(self, x: torch.Tensor, ax: int):
        """
        Extracts the forward slice along a specified axis.

        Args:
            x (torch.Tensor): Input tensor.
            ax (int): Axis along which to extract the forward slice.

        Returns:
            torch.Tensor: Forward slice.
        """
        return torch.narrow(x, ax, 2, x.shape[ax] - 2)
