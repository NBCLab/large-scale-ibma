"""Download NeuroVault data and convert to NiMARE dataset."""
import os.path as op
import os
import re
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


def replace_chars_with_dash(input_string):
    # Use regular expression to replace characters and symbols between words with '-'
    result_string = re.sub(r"\W+", "-", input_string)

    # Remove trailing dashes
    result_string = result_string.strip("-")

    return result_string


def get_nv_statisticmap(nv_data_dir):
    # Load the statistical images, which require merging several tables.
    # In order to merge `StatisticMap` to `Image`, we need the table `BaseStatisticMap`
    basecollectionitem = pd.read_csv(op.join(nv_data_dir, "statmaps_basecollectionitem.csv"))
    statisticmap = pd.read_csv(op.join(nv_data_dir, "statmaps_statisticmap.csv"))
    collection = pd.read_csv(op.join(nv_data_dir, "statmaps_collection.csv"))
    image = pd.read_csv(op.join(nv_data_dir, "statmaps_image.csv"))

    # `image` table is first merged to `basecollectionitem` using `basecollectionitem_ptr_id`:
    image_merged = pd.merge(
        image, basecollectionitem, left_on="basecollectionitem_ptr_id", right_on="id"
    )

    # Next, the `statisticmap` table can be merged to `image` using `image_ptr_id',
    # which corresponds to 'basecollectionitem_ptr_id'
    statisticmap_merged = pd.merge(
        statisticmap, image_merged, left_on="image_ptr_id", right_on="basecollectionitem_ptr_id"
    )

    # Finally, the `collection` table can be merged to `statisticmap` using `collection_id`,
    # which corresponds to `id` in `collection`
    return pd.merge(statisticmap_merged, collection, left_on="collection_id", right_on="id")


def get_task_name(task_id):
    cog_atlas_template_url = f"http://cognitiveatlas.org/api/v-alpha/task?id={task_id}"
    resp = requests.get(cog_atlas_template_url)
    return resp.json()["name"]


def mode_tie_break(df):
    result = pd.Series.mode(df)
    if len(result) > 1:
        return result[0]
    return result


def convert_to_nimare_dataset(necessary_image_info, contrast_name, img_dir):
    dataset_dict = {}
    for ii in necessary_image_info:
        dataset_dict[f"study-{ii['collection_id']}-{ii['id']}"] = {
            "contrasts": {
                contrast_name: {
                    "metadata": {"sample_sizes": None},
                    "images": {DEFAULT_MAP_TYPE_CONVERSION[ii["derived_map_type"]]: None},
                }
            }
        }
        (
            dataset_dict[f"study-{ii['collection_id']}-{ii['id']}"]["contrasts"][contrast_name][
                "images"
            ][DEFAULT_MAP_TYPE_CONVERSION[ii["derived_map_type"]]]
        ) = "/".join([img_dir, f"{ii['collection_id']}-{ii['id']}_{ii['fname']}"])

        if type(ii["sample_size"]) is int:
            dataset_dict[f"study-{ii['collection_id']}-{ii['id']}"]["contrasts"][contrast_name][
                "metadata"
            ]["sample_sizes"] = [ii["sample_size"]]
        else:
            print(f"\t\t{ii['sample_size']} not int", flush=True)

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


def derive_map_type(image_info):
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
                # map_type = "Other"
                continue
        else:
            map_type = ii["map_type"]

        if type(ii["number_of_subjects"]) is int:
            base_image_info.append(
                {
                    "id": ii["id"],
                    "collection_id": ii["collection_id"],
                    "derived_map_type": map_type,
                    "sample_size": ii["number_of_subjects"],
                    "fname": os.path.basename(ii["file"]),
                }
            )
        else:
            print(f'\t\t{ii["number_of_subjects"]}', flush=True)

    return base_image_info


def main(project_dir):
    NV_VERSION = "november_2022"

    data_dir = op.join(project_dir, "data")
    imgs_dir = op.join(data_dir, "images")
    nv_data_dir = op.join(data_dir, "nv-data", NV_VERSION)
    result_dir = op.join(project_dir, "results", "ibma")

    statisticmap_df = get_nv_statisticmap(nv_data_dir)

    # Select fMRI-BOLD maps with cognitive paradigm information
    fmri_statmaps = statisticmap_df.loc[statisticmap_df["modality"] == "fMRI-BOLD"]
    cogatlas_statmaps = fmri_statmaps.loc[~fmri_statmaps["cognitive_paradigm_cogatlas_id"].isna()]

    collection_terms = (
        cogatlas_statmaps.groupby("collection_id")["cognitive_paradigm_cogatlas_id"]
        .agg(mode_tie_break)
        .value_counts()
    )

    enough_maps = collection_terms[collection_terms >= 10]

    collection_terms_df = pd.DataFrame(enough_maps)
    cogatlas_names = collection_terms_df.index.map(get_task_name)
    collection_terms_df["cogatlas_name"] = cogatlas_names
    tasks = collection_terms_df.index.to_list()

    sel_columns = ["name", "img_name", "description_x", "file", "collection_id", "image_ptr_id"]

    maps_count_dict = {"task": [], "map": [], "count": []}
    cogat_terms_dict = {"cogat_id": [], "cogat_org_nm": [], "cogat_nm": [], "n_images": []}
    for task in tasks:
        print(f"Processing {task}", flush=True)
        task_org_name = collection_terms_df.loc[task, "cogatlas_name"]
        task_name = replace_chars_with_dash(task_org_name)

        result_task_dir = op.join(result_dir, task_name)
        dset_fn = op.join(result_task_dir, f"{task_name}_dset-raw.pkl.gz")
        if not op.isfile(dset_fn):
            all_task_maps = cogatlas_statmaps.query(
                f'cognitive_paradigm_cogatlas_id == "{task}"'
                ' & analysis_level == "G"'
                ' & is_thresholded == "f"'
                ' & (map_type == "Z" | map_type == "Other" | map_type == "T")'
            )

            # Output directory for images
            imgs_task_dir = op.join(imgs_dir, task_name)
            os.makedirs(imgs_task_dir, exist_ok=True)

            na_count = 0
            necessary_image_info = []
            necessary_nonother_image_info = []
            for map_type in ["Z", "T", "Other"]:
                print(f"\tProcessing {map_type}", flush=True)
                maps_df = all_task_maps.loc[all_task_maps["map_type"] == map_type][sel_columns]
                image_ids = maps_df["image_ptr_id"].to_list()

                image_info_list = download_images(image_ids, imgs_task_dir)
                image_clean_info_list = derive_map_type(image_info_list)

                if map_type != "Other":
                    necessary_nonother_image_info.extend(image_clean_info_list)

                necessary_image_info.extend(image_clean_info_list)

                maps_count_dict["task"].append(task_name)
                maps_count_dict["map"].append(map_type)
                maps_count_dict["count"].append(len(image_clean_info_list))

                na_count = na_count + len(image_ids) - len(image_clean_info_list)

            maps_count_dict["task"].append(task_name)
            maps_count_dict["map"].append("Not Available")
            maps_count_dict["count"].append(na_count)

            dset = convert_to_nimare_dataset(necessary_image_info, task_name, imgs_task_dir)
            z_dset = ImageTransformer("z").transform(dset)

            if len(z_dset.images) > 10:
                os.makedirs(result_task_dir, exist_ok=True)
                z_dset.save(dset_fn)

                cogat_terms_dict["cogat_id"].append(task)
                cogat_terms_dict["cogat_org_nm"].append(task_org_name)
                cogat_terms_dict["cogat_nm"].append(task_name)
                cogat_terms_dict["n_images"].append(len(z_dset.images))
        else:
            z_dset = Dataset.load(dset_fn)
            if len(z_dset.images) > 10:
                cogat_terms_dict["cogat_id"].append(task)
                cogat_terms_dict["cogat_org_nm"].append(task_org_name)
                cogat_terms_dict["cogat_nm"].append(task_name)
                cogat_terms_dict["n_images"].append(len(z_dset.images))

    maps_count_fn = op.join(data_dir, "maps_count.csv")
    if not op.isfile(maps_count_fn):
        maps_count_df = pd.DataFrame(maps_count_dict)
        maps_count_df.to_csv(maps_count_fn)

    cogat_terms_fn = op.join(data_dir, "cogat_terms.csv")
    if not op.isfile(cogat_terms_fn):
        cogat_terms_df = pd.DataFrame(cogat_terms_dict)
        cogat_terms_sorted_df = cogat_terms_df.sort_values(by="n_images", ascending=False)
        cogat_terms_sorted_df.to_csv(cogat_terms_fn, index=False)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
