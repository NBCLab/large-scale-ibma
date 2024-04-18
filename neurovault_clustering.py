import argparse
import os.path as op
import itertools
import os

import umap
import umap.plot
import pandas as pd
from nilearn import datasets
from sklearn.metrics.pairwise import cosine_similarity
from nilearn.maskers import (
    MultiNiftiMapsMasker,
    NiftiLabelsMasker,
    NiftiSpheresMasker,
)
from sklearn.metrics import silhouette_score
from nimare.dataset import Dataset
import hdbscan
import nibabel as nib
from nilearn._utils.niimg_conversions import _check_same_fov
from nilearn.image import concat_imgs, resample_to_img
import numpy as np

from utils import _exclude_outliers


def _get_parser():
    parser = argparse.ArgumentParser(description="Get clusters in NeuroVault data")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    parser.add_argument(
        "--atlas",
        dest="atlas",
        required=False,
        help="Name of the atlas to use for feature extraction",
    )
    parser.add_argument(
        "--n_cores",
        dest="n_cores",
        default=1,
        required=False,
        help="CPUs",
    )
    return parser


def _rm_nonstat_maps(dset, data_df):
    """
    Remove non-statistical maps from a dataset.
    """
    ids_to_keep = []
    for _, row in data_df.iterrows():
        image_name = row["image_name"]

        exclude = False
        for term in ["ICA", "PCA", "PPI"]:
            if term in image_name:
                exclude = True
                break

        if not exclude:
            ids_to_keep.append(row["id"])

    return dset.slice(ids_to_keep)


def _get_dset_features(dset, atlas=None):
    """
    Get the features of a dataset.
    """
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

    if atlas == "difumo":
        difumo = datasets.fetch_atlas_difumo(dimension=64, resolution_mm=2, legacy_format=False)
        masker_difumo = MultiNiftiMapsMasker(maps_img=difumo.maps, standardize="zscore_sample")

        data = masker_difumo.fit_transform(img4d)
    elif atlas == "schaefer":
        schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
        masker_schaefer = NiftiLabelsMasker(labels_img=schaefer.maps, standardize="zscore")

        data = masker_schaefer.fit_transform(img4d)
    elif atlas == "smith":
        smith = datasets.fetch_atlas_smith_2009()
        masker_smith = NiftiLabelsMasker(labels_img=smith["rsn20"], standardize="zscore")

        data = masker_smith.fit_transform(img4d)
    elif atlas == "power":
        power = datasets.fetch_coords_power_2011()
        coords = np.vstack((power.rois["x"], power.rois["y"], power.rois["z"])).T

        masker_power = NiftiSpheresMasker(seeds=coords, radius=10, standardize="zscore")

        data = masker_power.fit_transform(img4d)
    else:
        data = masker.transform(img4d)

    return data


def main(project_dir, atlas=None, n_cores=1):
    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results")
    clustering_dir = op.join(results_dir, "clustering")
    n_cores = int(n_cores)
    os.makedirs(clustering_dir, exist_ok=True)

    dset = Dataset.load(op.join(results_dir, "neurovault_full_dataset.pkl"))
    nv_collections_images_df = pd.read_csv(op.join(data_dir, "nv_collections_images.csv"))

    # TODO: This is a temporary fix to get the id of the images
    nv_collections_images_df["contrast_id"] = nv_collections_images_df.apply(
        lambda row: f"{row['collection_id']}-{row['image_id']}-nv", axis=1
    )
    nv_collections_images_df["id"] = nv_collections_images_df.apply(
        lambda row: f"{row['pmid']}-{row['contrast_id']}", axis=1
    )

    # For now, we will only use the following tasks
    sel_task = [
        "trm_4f2453ce33f16",
        "tsk_Ncknr0soiM4IV",
        "tsk_4a57abb949e8a",
        "tsk_mFS3uwUMAhXxe",
        "tsk_4a57abb949a93",
        "trm_4f23fc8c42d28",
    ]
    nv_collections_images_task_df = nv_collections_images_df[
        nv_collections_images_df["cognitive_paradigm_cogatlas_id"].isin(sel_task)
    ]

    dset = _rm_nonstat_maps(dset, nv_collections_images_df)
    dset_sel = dset.slice(nv_collections_images_task_df["id"].values)
    dset_sel_clean = _exclude_outliers(dset_sel)

    # Get features from dset given an atlas
    data = _get_dset_features(dset_sel_clean)

    # Compute QC matrices
    corr_data = np.corrcoef(data, rowvar=True)
    affinity_data = cosine_similarity(data)
    distance_data = affinity_data.max() - affinity_data
    np.save(op.join(clustering_dir, f"{atlas}_cor.npy"), corr_data)
    np.save(op.join(clustering_dir, f"{atlas}_aff.npy"), affinity_data)
    np.save(op.join(clustering_dir, f"{atlas}_dis.npy"), distance_data)

    # Initialize DataFrame to store results
    data_df = pd.DataFrame()
    data_df["id"] = dset_sel_clean.images["id"]
    data_df = pd.merge(data_df, nv_collections_images_df, how="left", on="id")
    data_df["pmid"] = data_df["pmid"].astype(str)

    # Set up dimensionality reduction and clustering parameters
    # n_neighbors range from 2 to a quarter of the data
    # 0.0 is the recommended value for finding clusters (smallest min_dist)
    components = range(2, 7)
    neighbors = np.linspace(5, 100, 5, dtype=int)
    distances = [0.0, 0.1, 0.25, 0.5, 0.8]
    samples = np.linspace(2, 30, 5, dtype=int)
    silhouette_dict = {"min_samples": [], "params": [], "score": []}
    for n_components, n_neighbors, min_dist in itertools.product(components, neighbors, distances):
        reduce_nv = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric="correlation",
            n_jobs=n_cores,
        )
        mapper_nv = reduce_nv.fit(data)
        embedding_nv = mapper_nv.transform(data)

        columns = [
            f"c-{n_components}-{i}_n-{n_neighbors}_d-{min_dist}" for i in range(n_components)
        ]
        embeding_df = pd.DataFrame(embedding_nv, columns=columns)
        data_df = pd.concat([data_df, embeding_df], axis=1)

        print(n_components, n_neighbors, min_dist)
        for min_samples in samples:
            print(min_samples)
            labels = hdbscan.HDBSCAN(
                min_samples=min_samples,
                min_cluster_size=10,
            ).fit_predict(embedding_nv)

            data_df[f"hdbscan_c-{n_components}_n-{n_neighbors}_d-{min_dist}_{min_samples}"] = (
                labels
            )
            data_df[f"hdbscan_c-{n_components}_n-{n_neighbors}_d-{min_dist}_{min_samples}"] = (
                data_df[
                    f"hdbscan_c-{n_components}_n-{n_neighbors}_d-{min_dist}_{min_samples}"
                ].astype(str)
            )
            silhouette = silhouette_score(embedding_nv, labels)

            silhouette_dict["min_samples"].append(min_samples)
            silhouette_dict["params"].append(f"c-{n_components}_n-{n_neighbors}_d-{min_dist}")
            silhouette_dict["score"].append(silhouette)

    silhouette_df = pd.DataFrame(silhouette_dict)
    silhouette_df.to_csv(op.join(clustering_dir, f"{atlas}_silhouette.csv"))
    data_df.to_csv(op.join(clustering_dir, f"{atlas}_data.csv"))


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
