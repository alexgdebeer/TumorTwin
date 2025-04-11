from typing import Optional, TypeVar

import numpy as np

from tumortwin.types.imaging import Image3D, Unit

CELLULARITY_NON_ENHANCE = 0.16  # Assigned from literature

T = TypeVar("T", bound=Image3D)


def ADC_to_cellularity(
    ADC: T,
    roi_enhance: Image3D,
    roi_nonenhance: Optional[Image3D] = None,
) -> T:
    """
    Converts an ADC map to a cellularity map using region of interest (ROI) masks.

    This function calculates the cellularity for enhancing and non-enhancing regions
    based on the apparent diffusion coefficient (ADC) values. Cellularity for
    non-enhancing regions is set to a fixed literature-based constant.

    Args:
        ADC (Image3D): The measured apparent diffusion coefficient (ADC) map.
        roi_enhance (Image3D): Binary mask for enhancing regions (e.g., tumors).
        roi_nonenhance (Image3D): Binary mask for non-enhancing regions.

    Returns:
        Image3D: The computed cellularity map, normalized between 0 and 1.

    Raises:
        AssertionError: If input images have mismatched shape or spacing.
    """

    # Check for units of ADC
    log10_array = np.log10(np.maximum(ADC.array, 1e-10))
    max_log10 = np.max(log10_array)

    # Set ADCW as a scalar based on the conditions
    if max_log10 > 3:
        ADCW = 3000
    elif 2 < max_log10 <= 3:
        ADCW = 300
    elif 1 < max_log10 <= 2:
        ADCW = 30
    elif 0 < max_log10 <= 1:
        ADCW = 3
    elif -1 < max_log10 <= 0:
        ADCW = 0.3
    elif -2 < max_log10 <= -1:
        ADCW = 0.03
    elif -3 < max_log10 <= -2:
        ADCW = 0.003
    else:
        ADCW = 3

    if roi_nonenhance is None:
        roi_nonenhance = roi_enhance.__class__.from_array(
            np.zeros_like(roi_enhance.array), roi_enhance
        )

    assert (
        ADC.shape == roi_enhance.shape == roi_nonenhance.shape
    ), "Input images have mismatched shape!"
    assert (
        ADC.spacing == roi_enhance.spacing == roi_nonenhance.spacing
    ), "Input images have mismatched spacing!"

    # Compute normalized cellularity for enhancing regions
    N_total = np.abs((ADCW - ADC.array) / (ADCW - 0))
    N_out = np.zeros(ADC.shape)

    # Assign cellularity values
    N_out[roi_nonenhance.array == 1] = CELLULARITY_NON_ENHANCE
    N_out[roi_enhance.array == 1] = N_total[roi_enhance.array == 1]

    # Clip cellularity values between 0 and 1
    return ADC.__class__.from_array(np.clip(N_out, 0.0, 1.0), ADC)


def compute_carrying_capacity(
    brain_mask: Image3D, cell_size_mm: float = 1e-6, packing_fraction: float = 0.75
) -> float:
    """
    Computes the physical carrying capacity (maximum number of cells) for a voxel.

    The carrying capacity is calculated based on the voxel volume, a packing fraction,
    and the average size of a cell.

    Args:
        brain_mask (Image3D): Binary mask of the brain region.
        cell_size_mm (float): Average cell size in mm³. Default is 1e-6 mm³.
        packing_fraction (float): Fraction of voxel space occupied by cells.
            Default is 0.75.

    Returns:
        float: The carrying capacity for a single voxel (cells per voxel).

    Raises:
        ValueError: If the unit of the brain mask spacing is invalid.
    """
    match brain_mask.spacing.unit:
        case Unit.MICRON:
            volume_units_scale_factor = 1.0e-3**3
        case Unit.MILLIMETER:
            volume_units_scale_factor = 1.0
        case Unit.METER:
            volume_units_scale_factor = 1.0e3**3
        case Unit.UNKNOWN:
            print("Warning: Units of input image are UNKNOWN. Defaulting to mm.")
            volume_units_scale_factor = 1.0
        case _:
            raise ValueError("Units of input image are not a valid length!")

    return (
        volume_units_scale_factor
        * brain_mask.spacing.x
        * brain_mask.spacing.y
        * brain_mask.spacing.z
        * packing_fraction
        / cell_size_mm
    )
