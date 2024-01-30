"""Utility functions for the meta-analysis pipeline."""
import numpy as np
import nibabel as nib
from nilearn._utils.niimg_conversions import _check_same_fov
from nilearn.image import concat_imgs, resample_to_img


def _get_outliers(data, ids):
    # Here n_voxels is data.shape[1] considering the nonaggresive mask is used
    n_studies, n_voxels = data.shape

    mask = ~np.isnan(data) & (data != 0)
    maps_lst = [data[i][mask[i]] for i in range(n_studies)]

    # Get outliers based on variance within each study
    # Set var of empty maps to 0, it will be remove by perc_outliers
    scores = [np.var(map_) if len(map_) > 1 else 0 for map_ in maps_lst]
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    threshold = 1.5 * iqr
    var_outliers = np.where((scores < q1 - threshold) | (scores > q3 + threshold))[0]

    # Get images with all positive or all negative values
    oth_outliers = np.array(
        [
            map_i
            for map_i, map_ in enumerate(maps_lst)
            if ((map_ > 0).sum() == len(map_)) or ((map_ < 0).sum() == len(map_))
        ],
        dtype=int,
    )

    # Get images with less than 40% of voxels
    perc_voxs = mask.sum(axis=1) / n_voxels
    perc_outliers = np.where(perc_voxs < 0.4)[0]

    outliers_idxs = np.unique(np.hstack([var_outliers, oth_outliers, perc_outliers]))

    return ids[outliers_idxs]


def _exclude_outliers(dset):
    # defaults for resampling images (nilearn's defaults do not work well)
    _resample_kwargs = {"clip": True, "interpolation": "linear"}

    images = dset.get_images(imtype="z")
    masker = dset.masker

    imgs = [
        (
            nib.load(img)
            if _check_same_fov(nib.load(img), reference_masker=masker.mask_img)
            else resample_to_img(nib.load(img), masker.mask_img, **_resample_kwargs)
        )
        for img in images
    ]

    img4d = concat_imgs(imgs, ensure_ndim=4)
    data = masker.transform(img4d)

    outliers = _get_outliers(data, dset.ids)
    unique_ids = np.setdiff1d(dset.ids, outliers)

    return dset.slice(unique_ids)
