import matplotlib.pyplot as plt

from tumortwin.types.base import BasePatientData


def plot_patient_timeline(patient_data: BasePatientData) -> None:
    """
    Generates a timeline plot for a given patient data object, showing imaging (visits) as vertical lines
    and treatment (chemotherapy + radiotherapy) as dosage over time.

    Args:
        patient_data (BasePatientData): The patient data object.
    """
    fig, ax1 = plt.subplots(figsize=(6, 2))

    # Extract timeline events
    visit_days = patient_data.visit_days
    chemotherapy_days = patient_data.chemotherapy_days
    radiotherapy_days = patient_data.radiotherapy_days

    # Extract dosages
    chemotherapy_doses = [c.dose for c in patient_data.chemotherapy]
    radiotherapy_doses = [r.dose for r in patient_data.radiotherapy]
    rt_units = (
        patient_data.radiotherapy[0].units.value
        if len(patient_data.radiotherapy) > 0
        else None
    )
    ct_units = (
        patient_data.chemotherapy[0].units.value
        if len(patient_data.chemotherapy) > 0
        else None
    )
    # Plot Visits (Imaging) as vertical lines
    for day in visit_days:
        ax1.axvline(
            x=day,
            color="blue",
            linestyle="--",
            alpha=0.5,
            label="Imaging visit" if day == visit_days[0] else "",
        )
    # Plot Radiotherapy as a line plot on the first y-axis
    if radiotherapy_days:
        ax1.plot(
            radiotherapy_days,
            radiotherapy_doses,
            marker="s",
            linestyle="-",
            color="green",
            label="Radiotherapy dose",
        )

    # Create a secondary y-axis for Radiotherapy
    ax2 = ax1.twinx()

    # Plot Chemotherapy as a line plot on the secondary y-axis
    if chemotherapy_days:
        ax2.plot(
            chemotherapy_days,
            chemotherapy_doses,
            marker="o",
            linestyle=" ",
            color="red",
            label="Chemotherapy dose",
        )

    # Formatting
    ax1.set_xlabel("Days since first visit")
    ax2.set_ylabel(
        f"Chemotherapy dose ({ct_units})",
        color="red",
    )
    ax1.set_ylabel(
        f"Radiotherapy dose ({rt_units})",
        color="green",
    )
    ax1.set_title(
        f"Patient {patient_data.patient} treatment timeline",
    )

    # Adding legends for both y-axes
    ax1.legend(loc="upper left")
    ax2.legend(loc="lower right")

    # for ax in [ax1, ax2]:
    # ax.tick_params(axis="both", labelsize=12)  # Adjust size
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    # label.set_fontname("Times New Roman")  # Set font to Times New Roman

    # Add gridlines and make sure it affects only the x-axis
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    # Show the plot
    plt.show()
