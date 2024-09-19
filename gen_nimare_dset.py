import argparse
import os.path as op

import pandas as pd
from nimare.dataset import Dataset
from nimare.io import DEFAULT_MAP_TYPE_CONVERSION
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


def convert_to_nimare_dataset(images_df, text_df, img_dir, suffix=""):
    suffix = f"-{suffix}" if suffix else ""

    images_df["pmid"] = images_df["pmid"].astype(int).astype(str)

    dataset_dict = {}
    for _, image in images_df.iterrows():
        id_ = image["pmid"]
        image_id = image["image_id"]
        collection_id = image["collection_id"]
        map_type = f"{image['map_type']} map"
        new_contrast_name = f"{collection_id}-{image_id}" + suffix

        if id_ not in dataset_dict:
            dataset_dict[id_] = {}

        if "contrasts" not in dataset_dict[id_]:
            dataset_dict[id_]["contrasts"] = {}

        text_df_row = text_df[text_df["pmid"] == int(id_)]
        if text_df_row.shape[0] == 0:
            title, keywords, abstract, body = None, None, None, None
        else:
            title = text_df_row["title"].values[0]
            keywords = text_df_row["keywords"].values[0]
            abstract = text_df_row["abstract"].values[0]
            body = text_df_row["body"].values[0]

        dataset_dict[id_]["contrasts"][new_contrast_name] = {
            "metadata": {
                "sample_sizes": [image["number_of_subjects"]],
                "pmid": image["pmid"],
                "pmcid": image["pmcid"],
                "collection_id": collection_id,
                "image_id": image_id,
                "map_type": image["map_type"],
                "cognitive_paradigm_cogatlas_id": image["cognitive_paradigm_cogatlas_id"],
                "cognitive_contrast_cogatlas_id": image["cognitive_contrast_cogatlas_id"],
                "contrast_definition": image["contrast_definition"],
                "cognitive_paradigm_cogatlas_name": image["cognitive_paradigm_cogatlas_name"],
                "cognitive_contrast_cogatlas_name": image["cognitive_contrast_cogatlas_name"],
                "image_name": image["image_name"],
                "image_file": image["image_file"],
            },
            "text": {
                "title": title,
                "keywords": keywords,
                "abstract": abstract,
                "body": body,
            },
            "images": {
                DEFAULT_MAP_TYPE_CONVERSION[map_type]: op.join(img_dir, image["image_path"])
            },
        }

    return Dataset(dataset_dict)


def main(project_dir):
    data_dir = op.join(project_dir, "data")
    image_dir = op.join(data_dir, "neurovault", "images")

    nv_collections_images_df = pd.read_csv(op.join(data_dir, "nv_all_collections_images.csv"))
    nv_text_df = pd.read_csv(op.join(data_dir, "pmid_text.csv"))
    dset_nv_fn = op.join(data_dir, "neurovault_all_dataset.pkl")

    print(f"Creating full dataset {nv_collections_images_df.shape[0]}", flush=True)
    dset_nv = convert_to_nimare_dataset(
        nv_collections_images_df,
        nv_text_df,
        image_dir,
    )
    dset_nv = ImageTransformer("z").transform(dset_nv)
    dset_nv = ImageTransformer("t").transform(dset_nv)
    dset_nv.save(dset_nv_fn)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
