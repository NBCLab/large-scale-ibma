import argparse
import os.path as op
import os

import pandas as pd
from nimare.io import DEFAULT_MAP_TYPE_CONVERSION
from nimare.dataset import Dataset
from nimare.transforms import ImageTransformer


def _get_parser():
    parser = argparse.ArgumentParser(description="Generate NiMARE dataset from NeuroVault data")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    return parser


def convert_to_nimare_dataset(images_df, contrast_name, img_dir):
    dataset_dict = {}
    for _, image in images_df.iterrows():
        collection_id = image["collection_id"]
        image_id = image["image_id"]
        image_path = image["image_path"]
        map_type = f"{image['map_type']} map"
        sample_size = image["number_of_subjects"]
        id_ = image["pmid"]
        new_contrast_name = f"{collection_id}-{image_id}-{contrast_name}"

        if id_ not in dataset_dict:
            dataset_dict[id_] = {}

        if "contrasts" not in dataset_dict[id_]:
            dataset_dict[id_]["contrasts"] = {}

        dataset_dict[id_]["contrasts"][new_contrast_name] = {
            "metadata": {"sample_sizes": None},
            "images": {DEFAULT_MAP_TYPE_CONVERSION[map_type]: None},
        }

        (
            dataset_dict[id_]["contrasts"][new_contrast_name]["images"][
                DEFAULT_MAP_TYPE_CONVERSION[map_type]
            ]
        ) = "/".join([img_dir, image_path])

        if type(sample_size) is int:
            dataset_dict[id_]["contrasts"][new_contrast_name]["metadata"]["sample_sizes"] = [
                sample_size
            ]
        else:
            print(f"\t\t{sample_size} not int", flush=True)

    print(dataset_dict)
    return Dataset(dataset_dict)


def main(project_dir):
    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results")
    cogat_res_dir = op.join(results_dir, "cogat_test")
    image_dir = op.join(data_dir, "nv-data", "images")
    os.makedirs(cogat_res_dir, exist_ok=True)

    # Select all images from the Go/No-Go task
    nv_collections_images_df = pd.read_csv(op.join(data_dir, "nv_collections_images.csv"))
    dset_nv_fn = op.join(cogat_res_dir, "neurovault_full_dataset.pkl")

    if not op.exists(dset_nv_fn):
        print(f"Creating full dataset {nv_collections_images_df.shape[0]}", flush=True)
        dset_nv = convert_to_nimare_dataset(
            nv_collections_images_df,
            "nv",
            image_dir,
        )
        z_dset_nv = ImageTransformer("z").transform(dset_nv)
        z_dset_nv.save(dset_nv_fn)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
