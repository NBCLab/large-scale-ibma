"""Download NeuroVault data and convert to NiMARE dataset."""

import os.path as op
import os
import argparse
import requests

from nimare.transforms import ImageTransformer
import nibabel as nib
from nimare.dataset import Dataset
import pandas as pd
from nimare.io import DEFAULT_MAP_TYPE_CONVERSION


def _get_parser():
    parser = argparse.ArgumentParser(description="Download NeuroVault data")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    return parser


def mode_tie_break(df):
    result = pd.Series.mode(df)
    if len(result) > 1:
        return result[0]
    return result


def convert_to_nimare_dataset(necessary_image_info, img_dir):
    dataset_dict = {}
    for ii in necessary_image_info:
        # Initialize dataset_dict
        dataset_dict[ii["pmid"]] = {
            "contrasts": {
                ii["contrast_id"]: {
                    "metadata": {
                        "sample_sizes": None,
                        "nv_col_id": None,
                        "nv_img_id": None,
                        "pmc_id": None,
                        "cogatlas_id": None,
                    },
                    "images": {DEFAULT_MAP_TYPE_CONVERSION[ii["derived_map_type"]]: None},
                    "text": {"title": None, "keywords": None, "abstract": None, "body": None},
                }
            }
        }

        # Add images
        (
            dataset_dict[ii["pmid"]]["contrasts"][ii["contrast_id"]]["images"][
                DEFAULT_MAP_TYPE_CONVERSION[ii["derived_map_type"]]
            ]
        ) = "/".join([img_dir, f"{ii['collection_id']}-{ii['image_id']}_{ii['fname']}"])

        # Add metadata
        if type(ii["sample_size"]) is int:
            dataset_dict[ii["pmid"]]["contrasts"][ii["contrast_id"]]["metadata"][
                "sample_sizes"
            ] = [ii["sample_size"]]
        else:
            print(f"\t\t{ii['sample_size']} not int", flush=True)

        dataset_dict[ii["pmid"]]["contrasts"][ii["contrast_id"]]["metadata"]["nv_col_id"] = ii[
            "collection_id"
        ]
        dataset_dict[ii["pmid"]]["contrasts"][ii["contrast_id"]]["metadata"]["nv_img_id"] = ii[
            "image_id"
        ]
        dataset_dict[ii["pmid"]]["contrasts"][ii["contrast_id"]]["metadata"]["pmc_id"] = ii[
            "pmcid"
        ]
        dataset_dict[ii["pmid"]]["contrasts"][ii["contrast_id"]]["metadata"]["cogatlas_id"] = ii[
            "cogatlas_id"
        ]

        # Add text data
        dataset_dict[ii["pmid"]]["contrasts"][ii["contrast_id"]]["text"]["title"] = ii["title"]
        dataset_dict[ii["pmid"]]["contrasts"][ii["contrast_id"]]["text"]["keywords"] = ii[
            "keywords"
        ]
        dataset_dict[ii["pmid"]]["contrasts"][ii["contrast_id"]]["text"]["abstract"] = ii[
            "abstract"
        ]
        dataset_dict[ii["pmid"]]["contrasts"][ii["contrast_id"]]["text"]["body"] = ii["body"]

    return Dataset(dataset_dict)


def download_images(image_ids, output_directory):
    image_info_list = []

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for image_id in image_ids:
        # Construct the NeuroVault API URL for image info
        image_info_url = f"https://neurovault.org/api/images/{image_id}/"

        try:
            # Make a GET request to fetch image info
            response = requests.get(image_info_url)

            if response.status_code == 200:
                image_info = response.json()

                # Download the image file
                image_url = image_info["file"]
                collection_id = image_info["collection_id"]
                image_filename = os.path.basename(image_url)
                image_path = os.path.join(
                    output_directory, f"{collection_id}-{image_id}_{image_filename}"
                )
                if not os.path.exists(image_path):
                    # Download the image
                    response = requests.get(image_url)
                    with open(image_path, "wb") as image_file:
                        image_file.write(response.content)

                try:
                    # Some image may be invalid
                    nib.load(image_path)
                except Exception:
                    pass
                else:
                    # Append image info to the list
                    image_info_list.append(image_info)

            # else:
            #     print(f"Failed to retrieve image info for ID {image_id}")

        except Exception as e:
            print(
                f"\t\tAn error occurred while processing image ID {image_id}: {str(e)}", flush=True
            )

    return image_info_list


def derive_map_type(image_info, maps_df):
    base_image_info = []
    for ii in image_info:
        if ii["map_type"] == "other":
            if "zstat" in ii["name"]:
                map_type = "Z map"
            elif "tstat" in ii["name"]:
                map_type = "T map"
            if "zstat" in ii["file"]:
                map_type = "Z map"
            elif "tstat" in ii["file"]:
                map_type = "T map"
            elif "Z_" in ii["description"]:
                map_type = "Z map"
            elif "T_" in ii["description"]:
                map_type = "T map"
            else:
                continue
        else:
            map_type = ii["map_type"]

        if type(ii["number_of_subjects"]) is int:
            sub_df = maps_df[maps_df["image_id"] == ii["id"]]
            pmids = sub_df["pmid"].values
            for pmid in pmids:
                sub_pmid_df = sub_df[sub_df["pmid"] == pmid]
                title = sub_pmid_df["title"].values[0]
                keywords = sub_pmid_df["keywords"].values[0]
                abstract = sub_pmid_df["abstract"].values[0]
                body = sub_pmid_df["body"].values[0]
                pmcid = sub_pmid_df["pmcid"].values[0]
                cogatlas_id = sub_pmid_df["cognitive_paradigm_cogatlas_id"].values[0]
                contrast_id = "1"  # I think there is not duplicate pmid
                base_image_info.append(
                    {
                        "pmid": pmid,
                        "contrast_id": contrast_id,
                        "title": title,
                        "keywords": keywords,
                        "abstract": abstract,
                        "body": body,
                        "pmcid": pmcid,
                        "cogatlas_id": cogatlas_id,
                        "collection_id": ii["collection_id"],
                        "image_id": ii["id"],
                        "derived_map_type": map_type,
                        "sample_size": ii["number_of_subjects"],
                        "fname": os.path.basename(ii["file"]),
                    }
                )

    return base_image_info


def main(project_dir):
    NV_VERSION = "november_2022"

    data_dir = op.join(project_dir, "data")
    imgs_dir = op.join(data_dir, "pubmed_images")
    nv_data_dir = op.join(data_dir, "nv-data", NV_VERSION)
    result_dir = op.join(project_dir, "results", "pubmed_ibma")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    dset_fn = op.join(result_dir, "pubmed_dataset-raw.pkl.gz")

    nv_pmid_df = pd.read_csv(op.join(nv_data_dir, "pmid_collection_statisticmaps_text_subset.csv"))

    na_count = 0
    maps_count_dict = {"map": [], "count": []}
    necessary_image_info = []
    necessary_nonother_image_info = []
    for map_type in ["Z", "T", "Other"]:
        print(f"\tProcessing {map_type}", flush=True)
        maps_df = nv_pmid_df.loc[nv_pmid_df["map_type"] == map_type]
        image_ids = maps_df["image_id"].to_list()

        # Loop over pmids since there are images with the same id but different pmids
        image_info_list = download_images(image_ids, imgs_dir)
        image_clean_info_list = derive_map_type(image_info_list, maps_df)

        if map_type != "Other":
            necessary_nonother_image_info.extend(image_clean_info_list)

        necessary_image_info.extend(image_clean_info_list)

        maps_count_dict["map"].append(map_type)
        maps_count_dict["count"].append(len(image_clean_info_list))

        na_count = na_count + len(image_ids) - len(image_clean_info_list)

    maps_count_dict["map"].append("Not Available")
    maps_count_dict["count"].append(na_count)

    maps_count_fn = op.join(data_dir, "pubmed_count.csv")
    maps_count_df = pd.DataFrame(maps_count_dict)
    maps_count_df.to_csv(maps_count_fn)

    dset = convert_to_nimare_dataset(necessary_image_info, imgs_dir)
    z_dset = ImageTransformer("z").transform(dset)
    z_dset.save(dset_fn)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
