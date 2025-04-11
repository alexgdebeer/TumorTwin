# import numpy as np

# from tumortwin.preprocessing.bound_condition_maker
# import bound_condition_maker
# from tumortwin.types.imaging import Image3D


# def test_bound_condition_maker(
#     brain_mask: Image3D, matlab_parameters: np.ndarray
# ) -> None:
#     bcf_original = matlab_parameters["bcf"]
#     bcf = bound_condition_maker(brain_mask)

#     assert bcf.shape == bcf_original.shape
#     matches = [
#         np.allclose(bcf.array[:, :, i], bcf_original[:, :, i])
#         for i in range(bcf.shape.z)
#     ]
#     assert np.sum(matches) == bcf.shape.z
