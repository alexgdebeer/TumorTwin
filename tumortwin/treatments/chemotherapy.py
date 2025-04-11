from datetime import datetime
from typing import List

import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib import pyplot as plt

from tumortwin.types import ChemotherapySpecification, TreatmentTime
from tumortwin.utils import days_since_first


def compute_chemo_concentration_for_dose(
    t: datetime,
    decay_rate: float,
    treatment_day: TreatmentTime,
    dose: float,
) -> float:
    """
    Compute the drug concentration for a single chemotherapy dose.

    This uses an exponential decay model to calculate the concentration of the drug
    at a specific time after its administration.

    Args:
        t (datetime): The current time for concentration calculation.
        decay_rate (float): The drug decay rate (1/day).
        treatment_day (TreatmentTime): The time the dose was administered.
        dose (float): The administered dose.

    Returns:
        float: The drug concentration at time `t`.
    """
    days = days_since_first(t, treatment_day)
    if days >= 0:
        return dose * np.exp(-decay_rate * days)
    else:
        return 0.0


def compute_chemo_concentrations(
    t: datetime, chemotherapy_specification: ChemotherapySpecification
) -> List[float]:
    """
    Compute the concentrations of chemotherapy drugs for all doses in a protocol.

    Args:
        t (datetime): The current time for concentration calculation.
        chemotherapy_specification (ChemotherapySpecification): Specifications
            for chemotherapy, including protocol and decay rate.

    Returns:
        List[float]: A list of drug concentrations, one for each dose in the protocol.
    """
    drug_concentrations_per_dose = [
        compute_chemo_concentration_for_dose(
            t, chemotherapy_specification.decay_rate, treatment_day, dose
        )
        for treatment_day, dose in chemotherapy_specification.protocol.items()
    ]
    return drug_concentrations_per_dose


def compute_cell_death_rate_for_chemo(
    t: datetime, chemo_spec: ChemotherapySpecification
) -> torch.Tensor:
    """
    Compute the rate of cell death induced by chemotherapy at a given time.

    Args:
        t (datetime): The current time for the calculation.
        chemo_spec (ChemotherapySpecification): Specifications for chemotherapy,
            including sensitivity and protocol.

    Returns:
        float: The rate of cell death induced by the drug.
    """
    return chemo_spec.sensitivity * torch.sum(
        torch.tensor(compute_chemo_concentrations(t, chemo_spec))
    )


def compute_total_cell_death_chemo(
    t: datetime, chemo_specs: List[ChemotherapySpecification]
) -> torch.Tensor:
    """
    Compute the total rate of cell death induced by multiple chemotherapy drugs.

    Args:
        t (datetime): The current time for the calculation.
        chemo_specs (List[ChemotherapySpecification]): A list of chemotherapy
            specifications for different drugs.

    Returns:
        float: The total rate of cell death induced by all drugs.
    """
    return torch.sum(
        torch.stack(
            [compute_cell_death_rate_for_chemo(t, spec) for spec in chemo_specs]
        )
    )


def plot_chemotherapy(
    timesteps: List[datetime],
    chemotherapy_specifications: List[ChemotherapySpecification],
) -> None:
    """
    Plot chemotherapy drug concentrations and their effects over time.

    This function visualizes the time-dependent concentrations of chemotherapy drugs
    and their sensitivity-scaled effects, as well as the total cell death rate.

    Args:
        timesteps (List[datetime]): The times at which to compute the concentrations and effects.
        chemotherapy_specifications (List[ChemotherapySpecification]): Specifications
            for the chemotherapy drugs to be plotted.
    """
    line_colors = list(mcolors.TABLEAU_COLORS.values())
    totals = np.zeros((len(timesteps),))
    _, ax = plt.subplots(len(chemotherapy_specifications) + 1, 1, figsize=(12, 12))

    for drug_idx, spec in enumerate(chemotherapy_specifications):
        concentrations_by_dose = [
            compute_chemo_concentrations(t, spec) for t in timesteps
        ]
        concentrations = np.array([np.sum(c) for c in concentrations_by_dose])

        # Plot individual doses
        for dose_idx in range(len(spec.protocol)):
            ax[drug_idx].plot(
                timesteps, [c[dose_idx] for c in concentrations_by_dose], "k"
            )

        # Plot total concentration for the drug
        ax[drug_idx].plot(
            timesteps,
            concentrations,
            color=line_colors[drug_idx],
            alpha=0.8,
            label=f"Total concentration of drug {drug_idx}",
        )
        ax[drug_idx].set_title(f"Chemotherapy drug {drug_idx}")
        ax[drug_idx].legend(loc="upper right")

        # Plot sensitivity-scaled concentration
        ax[-1].plot(
            timesteps,
            spec.sensitivity * concentrations,
            "--",
            color=line_colors[drug_idx],
            label=f"Sensitivity-scaled concentration of drug {drug_idx}",
        )
        totals += spec.sensitivity * concentrations

    # Plot total cell death rate
    ax[-1].legend(loc="upper right")
    ax[-1].set_title("Total effects")
    ax[-1].plot(timesteps, totals, "k", label="Total cell death rate")
    ax[-1].legend(loc="upper right")

    plt.xlabel("Time")
    plt.ylabel("Concentration / Effect")
    plt.tight_layout()
    plt.show()
