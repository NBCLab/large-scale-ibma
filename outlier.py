"""Outlier detection module."""

import os.path as op

import numpy as np
import pandas as pd
from nilearn._utils.niimg_conversions import check_same_fov
from nilearn.image import resample_to_img
from nimare.meta.utils import _apply_liberal_mask
from nimare.transforms import d_to_g, p_to_z, t_to_d
from nimare.utils import _boolean_unmask
from scipy.stats import linregress

from utils import get_data

PVAL = 0.05
ZMIN = p_to_z(PVAL, tail="two")
ZMAX = 50

TEMP_DIR = "/Users/julioaperaza/Documents/GitHub/large-scale-ibma/temp/outliers-hedges"


def _temp_check(
    robust_ave_data,
    robust_ave_signal,
    signal_mask,
    robust_ave_noise,
    noise_mask,
    masker,
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from nilearn.plotting import plot_stat_map

    robust_ave_signal_unmasked = _boolean_unmask(robust_ave_signal, signal_mask)
    robust_ave_signal_img = masker.inverse_transform(robust_ave_signal_unmasked)

    robust_ave_noise_unmasked = _boolean_unmask(robust_ave_noise, noise_mask)
    robust_ave_img = masker.inverse_transform(robust_ave_data)

    robust_ave_noise_img = masker.inverse_transform(robust_ave_noise_unmasked)

    fig = plt.figure(figsize=(10, 5))
    plot_stat_map(
        robust_ave_img,
        display_mode="mosaic",
        title="Median Image",
        cut_coords=5,
        figure=fig,
        output_file=op.join(TEMP_DIR, "01-robust_ave.png"),
    )

    ax = sns.displot(robust_ave_data.flatten(), bins=50, kde=True)
    ax.set(title="Distribution of the Median Image")
    plt.xlim(-0.2, 0.2)
    plt.savefig(op.join(TEMP_DIR, "02-robust_ave_dist.png"), bbox_inches="tight")

    fig = plt.figure(figsize=(10, 5))
    plot_stat_map(
        robust_ave_signal_img,
        display_mode="mosaic",
        title="Signal Map",
        cut_coords=5,
        figure=fig,
        output_file=op.join(TEMP_DIR, "03-signal_map.png"),
    )

    fig = plt.figure(figsize=(10, 5))
    plot_stat_map(
        robust_ave_noise_img,
        display_mode="mosaic",
        title="Noise Map",
        cut_coords=5,
        figure=fig,
        output_file=op.join(TEMP_DIR, "04-noise_map.png"),
    )

    ax = sns.displot(robust_ave_signal, bins=50, kde=True)
    ax.set(title="Signal Distribution")
    plt.xlim(-0.2, 0.2)
    plt.savefig(op.join(TEMP_DIR, "06-signal_dist.png"), bbox_inches="tight")

    ax = sns.displot(robust_ave_noise, bins=50, kde=True)
    ax.set(title="Noise Distribution")
    plt.xlim(-0.2, 0.2)
    plt.savefig(op.join(TEMP_DIR, "07-noise_dist.png"), bbox_inches="tight")


def _get_signal_mask(arr, pct=0.1):
    n = len(arr)
    idx_sorted = np.argsort(arr)  # Get sorted indices

    # Compute the indices for the percentiles
    bot_idx = int(np.floor(n * pct))  # 10th percentile
    top_idx = int(np.ceil(n * (1 - pct)))  # 90th percentile

    # Get the values for the bottom and top percentiles
    bot_idxs = idx_sorted[:bot_idx]
    top_idxs = idx_sorted[top_idx:]

    signal_mask = np.zeros(n, dtype=bool)
    signal_mask[bot_idxs] = True
    signal_mask[top_idxs] = True

    return signal_mask


def _get_noise_mask(arr, pct=0.2):
    """Select the bottom 20% of elements around zero"""
    n = len(arr)
    idx_sorted = np.argsort(np.abs(arr))  # Get sorted indices

    # Compute the indices for the percentiles
    bot_idx = int(np.floor(n * pct))  # 20th percentile

    # Get the values for the bottom and top percentiles
    bot_idxs = idx_sorted[:bot_idx]

    noise_mask = np.zeros(n, dtype=bool)
    noise_mask[bot_idxs] = True

    return noise_mask


def _get_iqr_outliers(scores):
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    threshold = 1.5 * iqr
    lower_bound = q1 - threshold
    upper_bound = q3 + threshold

    return (scores < lower_bound) | (scores > upper_bound)


def _rm_nonstat_maps(dset):
    """
    Remove non-statistical maps from a dataset.

    Notes
    -----
    This function requires the dataset to have a metadata field called
    "image_name" and "image_file".
    """
    new_dset = dset.copy()
    data_df = dset.metadata

    assert "image_name" in data_df.columns

    sel_ids = []
    for _, row in data_df.iterrows():
        image_name = row["image_name"].lower()
        file_name = row["image_file"].lower()

        exclude = False
        for term in ["ica", "pca", "ppi", "seed", "functional connectivity", "correlation"]:
            if term in image_name:
                exclude = True
                break

        if "cope" in file_name and ("zstat" not in file_name and "tstat" not in file_name):
            exclude = True

        if "tfce" in file_name:
            exclude = True

        if not exclude:
            sel_ids.append(row["id"])

    new_dset = new_dset.slice(sel_ids)
    new_dset.metadata = new_dset.metadata.reset_index()
    return new_dset


def _rm_outliers(dset):
    new_dset = dset.copy()
    data = get_data(dset, imtype="z")

    outliers_idxs = []
    for img_i, img in enumerate(data):
        max_val = np.max(img)
        min_val = np.min(img)

        # Catch any inverted p-value, effect size or correlation maps
        if max_val < ZMIN and min_val > -ZMIN:
            outliers_idxs.append(img_i)

        # Catch any map with extreme values
        if max_val > ZMAX or min_val < -ZMAX:
            outliers_idxs.append(img_i)

        # Catch any map with all positive or all negative values
        if ((img > 0).sum() == len(img)) or ((img < 0).sum() == len(img)):
            outliers_idxs.append(img_i)

    rm_ids = dset.ids[np.array(outliers_idxs, dtype=int)]
    sel_ids = np.setdiff1d(dset.ids, rm_ids)

    new_dset = new_dset.slice(sel_ids)
    new_dset.metadata = new_dset.metadata.reset_index()
    return new_dset


def find_inverted_contrast(scores):
    score_set = set(scores)

    inverted_contrast = []
    for score in scores:
        if score < 0:
            if -score in score_set:
                inverted_contrast.append(True)
            else:
                inverted_contrast.append(False)
        else:
            inverted_contrast.append(False)

    return np.array(inverted_contrast)


def _rm_outliers_basic(dset, target=None):
    new_dset = dset.copy()
    data = get_data(new_dset, imtype="z")

    if check_same_fov(target, reference_masker=dset.masker.mask_img):
        target_data = dset.masker.transform(target)
    else:
        target_data = dset.masker.transform(resample_to_img(target, dset.masker.mask_img))

    target_data = target_data.reshape(-1)
    correlations = np.array([np.corrcoef(row, target_data)[0, 1] for row in data])
    sel_ids = new_dset.ids[correlations > 0.4]

    return new_dset.slice(sel_ids)


def _rm_outliers_advanced(dset):
    new_dset = dset.copy()
    masker = new_dset.masker
    data = get_data(new_dset, imtype="t")
    n_studies, n_voxels = data.shape

    sample_sizes = np.array(
        [np.mean(sample_size) for sample_size in new_dset.metadata["sample_sizes"]]
    )
    n_maps = np.tile(sample_sizes, (n_voxels, 1)).T
    cohens_maps = t_to_d(data, n_maps)
    hedges_maps = d_to_g(cohens_maps, n_maps)

    data_bags = zip(*_apply_liberal_mask(hedges_maps))
    keys = ["values", "voxel_mask", "study_mask"]
    data_bags_dict = [dict(zip(keys, bag)) for bag in data_bags]

    # Calculate robust image
    robust_ave_data = np.zeros(n_voxels)
    for bag in data_bags_dict:
        values = bag["values"]
        voxel_mask = bag["voxel_mask"]
        study_mask = bag["study_mask"]

        # Get the average value for each voxel across studies
        robust_ave_data[voxel_mask] = np.median(values, axis=0)

    # Get signal and noise masks
    signal_mask = _get_signal_mask(robust_ave_data)
    noise_mask = _get_noise_mask(robust_ave_data)

    robust_ave_signal = robust_ave_data[signal_mask]
    robust_ave_noise = robust_ave_data[noise_mask]
    std_x = np.std(robust_ave_noise)

    _temp_check(
        robust_ave_data,
        robust_ave_signal,
        signal_mask,
        robust_ave_noise,
        noise_mask,
        masker,
    )
    robust_slopes = []
    signal_slope = []
    std_noise = []
    for img in hedges_maps:
        img_signal = img[signal_mask]
        img_noise = img[noise_mask]

        corr = np.corrcoef(img_signal, robust_ave_signal)[0, 1]
        std_y = np.std(img_noise)
        std_noise.append(std_y)
        robust_slopes.append(corr * std_y / std_x)
        signal_slope.append(linregress(robust_ave_signal, img_signal).slope)

    robust_slopes = np.array(robust_slopes)
    signal_slope = np.array(signal_slope)

    # Find inverted contrast, keep the ones with positive slope
    inverted_ids = find_inverted_contrast(robust_slopes)
    print("Inverted contrast:", inverted_ids.sum())
    print(new_dset.ids[inverted_ids])

    robust_slopes = robust_slopes[~inverted_ids]
    signal_slope = signal_slope[~inverted_ids]
    std_noise = np.array(std_noise)[~inverted_ids]

    new_dset = new_dset.slice(new_dset.ids[~inverted_ids])

    iqr_outliers = _get_iqr_outliers(signal_slope)

    data_df = pd.DataFrame(
        {
            "id": new_dset.ids,
            "slope": robust_slopes,
            "signal_slope": signal_slope,
            "noise_std": std_noise,
            "collection_id": new_dset.metadata["collection_id"],
            "image_id": new_dset.metadata["image_id"],
            "outlier": iqr_outliers,
        }
    )
    data_df.to_csv(op.join(TEMP_DIR, "data.csv"), index=False)

    return new_dset.slice(new_dset.ids[~iqr_outliers])


def remove_outliers(dset, method="full", target=None):
    # Remove non-statistical maps
    dset = _rm_nonstat_maps(dset)
    dset = _rm_outliers(dset)

    if method == "basic" or method == "full":
        dset = _rm_outliers_basic(dset, target=target)

    if method == "advanced" or method == "full":
        dset = _rm_outliers_advanced(dset)

    return dset
