import argparse
import os.path as op
import itertools
import os
from glob import glob

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

from utils import _exclude_outliers, _rm_nonstat_maps

# Set up dimensionality reduction and clustering parameters
# n_neighbors range from 2 to a quarter of the data
# 0.0 is the recommended value for finding clusters (smallest min_dist)
COMPONENTS = range(2, 7)
NEIGHBORS = np.linspace(5, 1000, 5, dtype=int)
DISTANCES = [0.0, 0.1, 0.25, 0.5, 0.8]
SAMPLES = np.linspace(2, 30, 5, dtype=int)  # min_samples for HDBSCAN


def _get_parser():
    parser = argparse.ArgumentParser(description="Get clusters in NeuroVault data")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    parser.add_argument(
        "--database",
        dest="database",
        required=False,
        help="Name of the database to use for extracting brain maps",
    )
    parser.add_argument(
        "--atlas",
        dest="atlas",
        required=False,
        help="Name of the atlas to use for feature extraction",
    )
    parser.add_argument(
        "--space",
        dest="space",
        required=False,
        help="Name of the space to use for feature extraction",
    )
    parser.add_argument(
        "--tasks",
        dest="tasks",
        required=False,
        nargs="+",
        help="Name of the space to use for feature extraction",
    )
    parser.add_argument(
        "--n_cores",
        dest="n_cores",
        default=1,
        required=False,
        help="CPUs",
    )
    return parser


def _get_features_from_imgs(images, masker, atlas=None):
    """
    Get the features of a dataset.
    """
    _resample_kwargs = {"clip": True, "interpolation": "linear"}

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
        masker_schaefer = NiftiLabelsMasker(labels_img=schaefer.maps, standardize="zscore_sample")

        data = masker_schaefer.fit_transform(img4d)
    elif atlas == "smith":
        smith = datasets.fetch_atlas_smith_2009()
        masker_smith = NiftiLabelsMasker(labels_img=smith["rsn20"], standardize="zscore_sample")

        data = masker_smith.fit_transform(img4d)
    elif atlas == "power":
        power = datasets.fetch_coords_power_2011()
        coords = np.vstack((power.rois["x"], power.rois["y"], power.rois["z"])).T

        masker_power = NiftiSpheresMasker(seeds=coords, radius=5, standardize="zscore_sample")

        data = masker_power.fit_transform(img4d)
    else:
        data = masker.transform(img4d)

    return data


def main(project_dir, database="neurovault", atlas=None, space="umap", tasks=None, n_cores=1):
    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results")
    clustering_dir = op.join(results_dir, f"{database}_clustering")
    n_cores = int(n_cores)
    os.makedirs(clustering_dir, exist_ok=True)

    dset = Dataset.load(op.join(data_dir, "neurovault_full_dataset.pkl"))
    masker = dset.masker

    # Initialize DataFrame to store results
    data_df = pd.DataFrame()
    data_fn = op.join(clustering_dir, f"{atlas}_dat.npy")

    # Calculating the data matrix is computationally expensive, see if it has been saved
    if not op.isfile(data_fn):
        if database == "neurovault":
            if tasks:
                id_sel = dset.metadata[
                    dset.metadata["cognitive_paradigm_cogatlas_id"].isin(tasks)
                ]["id"].values
                dset = dset.slice(id_sel)

            dset = _rm_nonstat_maps(dset)
            dset = _exclude_outliers(dset)

            data_df["id"] = dset.images["id"]
            data_df = pd.merge(data_df, dset.metadata, how="left", on="id")

            # Get features from dset given an atlas
            images = dset.get_images(imtype="z")

        elif database == "neurosynth":
            metamaps_dir = op.join(data_dir, "neurosynth", "metamaps")

            images = sorted(glob(os.path.join(metamaps_dir, "*.nii*")))
            features = [os.path.basename(img).split(os.extsep)[0] for img in images]

            data_df["feature"] = features
        else:
            raise ValueError("Invalid database")

        data = _get_features_from_imgs(images, masker, atlas=atlas)
        np.save(data_fn, data)
    else:
        data = np.load(data_fn)

    # Compute QC matrices
    corr_data = np.corrcoef(data, rowvar=True)
    affinity_data = cosine_similarity(data)
    distance_data = affinity_data.max() - affinity_data
    np.save(op.join(clustering_dir, f"{atlas}_cor.npy"), corr_data)
    np.save(op.join(clustering_dir, f"{atlas}_aff.npy"), affinity_data)
    np.save(op.join(clustering_dir, f"{atlas}_dis.npy"), distance_data)

    if space == "umap":
        # silhouette_dict = {"min_samples": [], "params": [], "score": []}
        for n_components, n_neighbors, min_dist in itertools.product(
            COMPONENTS, NEIGHBORS, DISTANCES
        ):
            print(n_components, n_neighbors, min_dist)

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

            for min_samples in SAMPLES:
                labels = hdbscan.HDBSCAN(
                    min_samples=min_samples,
                    min_cluster_size=10,
                ).fit_predict(embedding_nv)

                labels = labels.astype(str)
                data_df[f"hdbscan_c-{n_components}_n-{n_neighbors}_d-{min_dist}_{min_samples}"] = (
                    labels
                )

                # silhouette = silhouette_score(embedding_nv, labels)

                # silhouette_dict["min_samples"].append(min_samples)
                # silhouette_dict["params"].append(f"c-{n_components}_n-{n_neighbors}_d-{min_dist}")
                # silhouette_dict["score"].append(silhouette)

        # silhouette_df = pd.DataFrame(silhouette_dict)
        # silhouette_df.to_csv(op.join(clustering_dir, f"{atlas}_silhouette.csv"))
    elif space == "full":
        for min_samples in SAMPLES:
            labels = hdbscan.HDBSCAN(
                min_samples=min_samples,
                min_cluster_size=10,
            ).fit_predict(data)

            labels = labels.astype(str)
            data_df[f"hdbscan_{min_samples}"] = labels

    data_df.to_csv(op.join(clustering_dir, f"{space}-{atlas}_data.csv"))


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
