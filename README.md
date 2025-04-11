
[![License](https://img.shields.io/github/license/OncologyModelingGroup/TumorTwin)](./LICENSE)
[![Top language](https://img.shields.io/github/languages/top/OncologyModelingGroup/TumorTwin)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/)
[![Issues](https://img.shields.io/github/issues/OncologyModelingGroup/TumorTwin)](https://github.com/OncologyModelingGroup/TumorTwin/issues)

## ![TumorTwin Logo](./docs/assets/tumor_twin.png)
`TumorTwin` is a python framework for creating image-guided cancer-patient digital twins (CPDTs). It includes functionality for:
1. Loading and handling patient datasets, including treatment data and multi-modal longitudinal MRI data.
2. Simulating tumor growth and response to treatment.
3. Computing sensitivities of model output quantities of interest with respect to model parameters
4. Calibrating model parameters to longitudinal MRI data.

## Quick start
To jump right in, click the badge below to launch a tutorial/demo in Google Colab:

[![HGG Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OncologyModelingGroup/TumorTwin/blob/main/tutorials/HGG_Calibration.ipynb)

## Full Documentation

For installation instructions, tutorials, and full API reference check out out the [complete documentation here.](https://OncologyModelingGroup.github.io/TumorTwin)

## License

This package is released under a UT Austin Research license. Commercial use is prohibited without prior permission from UT Austin. See [LICENSE.md](https://OncologyModelingGroup.github.io/TumorTwin/LICENSE.md) for more details.

## References
If you find this library useful in your research, please consider citing the following references:

#### TumorTwin Technical Report
```
TBD
```
#### TumorTwin Codebase
```
@misc{tumortwin,
	author={Kapteyn, Michael G. and Chaudhuri, Anirban and Lima, Ernesto A.B.F. and Pash, Graham and Bravo, Rafael and Yankeelov, Thomas E. and Willcox, Karen and Hormuth II, David A.},
	title={TumorTwin},
	year={2025},
	url={https://github.com/OncologyModelingGroup/TumorTwin},
}
```
