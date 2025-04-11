import numpy as np

from tumortwin.types.imaging import Image3D


def compute_ccc(
    x: np.ndarray, y: np.ndarray, bias: bool = True, use_pearson: bool = False
) -> float:
    """
    Compute the concordance correlation coefficient between two arrays x and y.

    Args:
        x (np.ndarray): First vector.
        y (np.ndarray): Second vector.
        bias (bool, optional): Bias correction. Defaults to True.
        use_pearson (bool, optional): Use Pearson correlation coefficient. Defaults to False.

    Returns:
        float: The concordance correlation coefficient.
    """
    if use_pearson:
        cor = np.corrcoef(x, y)[0][1]  # Pearson correlation coefficient

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        var_x = np.var(x)
        var_y = np.var(y)
        sd_x = np.std(x)
        sd_y = np.std(y)

        numerator = 2 * cor * sd_x * sd_y
        denominator = var_x + var_y + (mean_x - mean_y) ** 2
        return numerator / denominator

    var_x, cov_xy, _, var_y = np.cov(x, y, bias=bias).flat
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = 2 * cov_xy
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    return numerator / denominator


def compute_voxel_ccc(
    x_img: Image3D,
    y_img: Image3D,
    roi_img: Image3D,
    bias: bool = True,
    use_pearson: bool = False,
) -> float:
    """
    Compute the voxel-wise concordance correlation coefficient between two NIfTI files.

    Args:
        x_img (Image3D): x image.
        y_img (Image3D): y image.
        roi_img (Image3D): ROI image.
        bias (bool, optional): Bias correction. Defaults to True.
        use_pearson (bool, optional): Use Pearson correlation coefficient. Defaults to False.

    Returns:
        float: Voxel-wise concordance correlation coefficient.
    """
    assert x_img.shape == y_img.shape, "Images must have the same shape."
    assert x_img.spacing == y_img.spacing, "Images must have the same voxel size."
    assert np.array_equal(
        x_img.centroid, y_img.centroid
    ), "Images must have the same centroid."
    assert x_img.shape == roi_img.shape, "ROI must have the same shape as images."
    assert x_img.spacing == roi_img.spacing, "ROI must have the same voxel size."
    assert np.array_equal(
        x_img.centroid, roi_img.centroid
    ), "ROI must have the same centroid."

    x_data = x_img.array
    y_data = y_img.array
    roi_mask = roi_img.array > 0

    x_roi = np.ma.masked_array(x_data, mask=~roi_mask)
    y_roi = np.ma.masked_array(y_data, mask=~roi_mask)

    x_vox = x_roi.compressed()
    y_vox = y_roi.compressed()

    return compute_ccc(x_vox, y_vox, bias, use_pearson)


def compute_dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Dice coefficient between y_true and y_pred.

    Args:
        y_true (np.ndarray): Ground truth mask.
        y_pred (np.ndarray): Predicted mask.

    Returns:
        float: The computed Dice coefficient.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)

    return (2.0 * intersection) / union


def compute_voxel_dice(x_img: Image3D, y_img: Image3D, threshold: float = 0.5) -> float:
    """
    Compute voxel-wise Dice coefficient between two NIfTI files.

    Args:
        x_img (Image3D): x image.
        y_img (Image3D): y image.
        threshold (float, optional): Value to threshold on. Defaults to 0.5.

    Returns:
        float: Dice coefficient.
    """
    assert x_img.shape == y_img.shape, "Images must have the same shape."
    assert x_img.spacing == y_img.spacing, "Images must have the same voxel size."
    assert np.array_equal(
        x_img.centroid, y_img.centroid
    ), "Images must have the same centroid."

    x_data = x_img.array
    y_data = y_img.array

    x_bin = (x_data > threshold).astype(int)
    y_bin = (y_data > threshold).astype(int)

    return compute_dice(x_bin, y_bin)


def compute_voxel_ttc(x_img: Image3D, carrying_capacity: float) -> float:
    """
    Compute voxel-wise total tumor cellularity (TTC) from a NIfTI file.

    Args:
        x_img (Image3D): Image3D object.
        carrying_capacity (float): Carrying capacity.

    Returns:
        float: Total tumor cellularity.
    """
    return carrying_capacity * np.sum(x_img.array)


def compute_voxel_ttv(x_img: Image3D, threshold: float = 0.5) -> float:
    """
    Compute voxel-wise total tumor volume (TTV) from a NIfTI file.

    Args:
        x_img (Image3D): Image3D object.
        threshold (float, optional): Threshold for tumor detection. Defaults to 0.5.

    Returns:
        float: Total tumor volume.
    """
    x_data = x_img.array
    x_bin = (x_data > threshold).astype(int)

    return np.sum(x_bin) * x_img.voxvol
