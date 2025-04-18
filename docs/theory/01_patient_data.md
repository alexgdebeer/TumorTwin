
# Patient Data & Imaging

TumorTwin relies on quantitative MRI data and treatment history to personalize digital twins.
Data are imported in the NIFTI format and structured into `PatientData` objects validated via Pydantic.

## Observation Function

To infer tumor cellularity from MRI, the apparent diffusion coefficient (ADC) is transformed using:

\[
N(x, t) = \frac{ADC_w - ADC(x, t)}{ADC(x, t) - ADC_{min}}
\]

where:
- \( N(x, t) \): tumor cellularity at voxel \( x \), time \( t \)
- \( ADC_w = 3.0 \times 10^{-3} \, \text{mm}^2/\text{s} \): water diffusivity
- \( ADC_{min} \): minimum ADC in tumor region

The derived cellularity maps serve as ground truth for calibration and prediction.
    