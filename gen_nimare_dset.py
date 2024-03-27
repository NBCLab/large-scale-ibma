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
        id_ = f"study-{collection_id}-{image_id}"

        dataset_dict[id_] = {
            "contrasts": {
                contrast_name: {
                    "metadata": {"sample_sizes": None},
                    "images": {DEFAULT_MAP_TYPE_CONVERSION[map_type]: None},
                }
            }
        }
        (
            dataset_dict[id_]["contrasts"][contrast_name]["images"][
                DEFAULT_MAP_TYPE_CONVERSION[map_type]
            ]
        ) = "/".join([img_dir, image_path])

        if type(sample_size) is int:
            dataset_dict[id_]["contrasts"][contrast_name]["metadata"]["sample_sizes"] = [
                sample_size
            ]
        else:
            print(f"\t\t{sample_size} not int", flush=True)

    return Dataset(dataset_dict)


def main(project_dir):
    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results")
    cogat_res_dir = op.join(results_dir, "cogat_test")
    image_dir = op.join(data_dir, "nv-data", "images")
    os.makedirs(cogat_res_dir, exist_ok=True)

    # Select all images from the Go/No-Go task
    nv_collections_images_df = pd.read_csv(op.join(data_dir, "nv_collections_images.csv"))
    nv_collections_images_gonogo_df = nv_collections_images_df[
        nv_collections_images_df.cognitive_paradigm_cogatlas_id == "tsk_4a57abb949a93"
    ]

    # Select one image per collection
    collections = nv_collections_images_gonogo_df.collection_id.unique()
    image_selected_lst = []
    for collection in collections:
        sub_df = nv_collections_images_df[nv_collections_images_df.collection_id == collection]
        image_selected = sub_df.sample(1)
        image_selected_lst.append(image_selected)
    nv_collections_images_gonogo_rand_df = pd.concat(image_selected_lst)

    # Select the most relevant image per collection
    sel_images = [
        28449,
        28869,
        29375,
        43883,
        51756,
        57054,
        65854,
        68578,
        127869,
        127949,
        129029,
        129184,
        191520,
        392361,
        550207,
        550241,
        550290,
    ]
    nv_collections_images_gonogo_sel_df = nv_collections_images_gonogo_df[
        nv_collections_images_gonogo_df.image_id.isin(sel_images)
    ]

    dset_fn = op.join(cogat_res_dir, "go-no-go-task_full_dataset.pkl")
    dset_rand_fn = op.join(cogat_res_dir, "go-no-go-task_random_dataset.pkl")
    dset_select_fn = op.join(cogat_res_dir, "go-no-go-task_selected_dataset.pkl")

    print(f"Creating full dataset {nv_collections_images_gonogo_df.shape[0]}", flush=True)
    dset = convert_to_nimare_dataset(nv_collections_images_gonogo_df, "go_no-go_task", image_dir)
    z_dset = ImageTransformer("z").transform(dset)
    z_dset.save(dset_fn)

    print(f"Creating random dataset {nv_collections_images_gonogo_rand_df.shape[0]}", flush=True)
    dset_rand = convert_to_nimare_dataset(
        nv_collections_images_gonogo_rand_df, "go_no-go_task", image_dir
    )
    z_dset_rand = ImageTransformer("z").transform(dset_rand)
    z_dset_rand.save(dset_rand_fn)

    print(f"Creating select dataset {nv_collections_images_gonogo_sel_df.shape[0]}", flush=True)
    dset_select = convert_to_nimare_dataset(
        nv_collections_images_gonogo_sel_df, "go_no-go_task", image_dir
    )
    z_dset_select = ImageTransformer("z").transform(dset_select)
    z_dset_select.save(dset_select_fn)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
