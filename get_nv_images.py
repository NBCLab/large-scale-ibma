"""Get NeuroVault images for collections linked to PubMed articles."""

import argparse
import os
import os.path as op

import nibabel as nib
import pandas as pd
import requests

NEUROSCOUT_OWNER_ID = 5761
NV_VERSION = "february_2024"

KEEP_IMG_COLUMNS = [
    "image_name",
    "map_type",
    "image_file",
    "collection_id",
    "image_id",
    "number_of_subjects",
    "cognitive_paradigm_cogatlas_id",
    "cognitive_contrast_cogatlas_id",
    "contrast_definition",
]

KEEP_COL_COLUMNS = [
    "pmid",
    "pmcid",
    "doi",
    "secondary_doi",
    "collection_id",
    "collection_name",
    "source",
]


def _get_parser():
    parser = argparse.ArgumentParser(description="Download NeuroVault data")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    return parser


def download_images(image_ids, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    image_info_dict = {"image_id": [], "image_path": []}
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
                rel_path = f"{collection_id}-{image_id}_{image_filename}"
                image_path = os.path.join(output_directory, rel_path)
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
                    image_info_dict["image_id"].append(image_id)
                    image_info_dict["image_path"].append(rel_path)

        except Exception as e:
            print(
                f"An error occurred while processing image ID {image_id}: {str(e)}",
                flush=True,
            )

    return image_info_dict


def derive_map_type(row):
    if row["map_type"] == "Other":
        row["description"] = str(row["description"])

        if "zstat" in row["name"]:
            map_type = "Z"
        elif "tstat" in row["name"]:
            map_type = "T"
        elif "zstat" in row["file"]:
            map_type = "Z"
        elif "tstat" in row["file"]:
            map_type = "T"
        elif "Z_" in row["description"]:
            map_type = "Z"
        elif "T_" in row["description"]:
            map_type = "T"
        else:
            map_type = None
    else:
        map_type = row["map_type"]

    return map_type


def main(project_dir):
    data_dir = op.join(project_dir, "data")
    nv_data_dir = op.join(data_dir, "neurovault", NV_VERSION)
    image_dir = op.join(data_dir, "neurovault", "images")
    os.makedirs(image_dir, exist_ok=True)

    basecollectionitem = pd.read_csv(op.join(nv_data_dir, "statmaps_basecollectionitem.csv"))
    image = pd.read_csv(op.join(nv_data_dir, "statmaps_image.csv"))
    statisticmap = pd.read_csv(op.join(nv_data_dir, "statmaps_statisticmap.csv"))
    cogat_tasks = pd.read_csv(op.join(nv_data_dir, "statmaps_cognitiveatlastask.csv"))
    cogat_contrasts = pd.read_csv(op.join(nv_data_dir, "statmaps_cognitiveatlascontrast.csv"))[
        ["name", "cog_atlas_id"]
    ]

    # Load the file with NeuroVault collections linked to PubMed articles
    # (created by get_nv_collections.py)
    collections_pmid_df = pd.read_csv(op.join(data_dir, "nv_all_collections.csv"))
    image_merged = pd.merge(
        image, basecollectionitem, left_on="basecollectionitem_ptr_id", right_on="id"
    )
    statisticmap_merged = pd.merge(
        statisticmap, image_merged, left_on="image_ptr_id", right_on="basecollectionitem_ptr_id"
    )

    # Remove rows with missing cognitive_paradigm_cogatlas_id and number_of_subjects
    # We need both to perform IBMA
    statisticmap_merged = statisticmap_merged.dropna(
        subset=["cognitive_paradigm_cogatlas_id", "number_of_subjects"]
    )

    # Keep only rows with collection_id in collections_pmid_df
    # Skip this for now, as we want to keep all collections for the Baseline model
    # statisticmap_merged = statisticmap_merged[
    #     statisticmap_merged.collection_id.isin(collections_pmid_df.collection_id)
    # ]

    # Filter the statisticmap_merged DataFrame
    statisticmap_filtered = statisticmap_merged.query(
        'modality == "fMRI-BOLD"'
        ' & analysis_level == "G"'
        ' & is_thresholded == "f"'
        ' & (map_type == "Z" | map_type == "Other" | map_type == "T")'
        " & brain_coverage > 40"
        " & number_of_subjects > 10"
        ' & cognitive_paradigm_cogatlas_id != "trm_4c8a834779883"'  # rest eyes open
        ' & cognitive_paradigm_cogatlas_id != "trm_54e69c642d89b"'  # rest eyes closed
        ' & not_mni == "f"'
    )

    # Relabel the "Other" map type into "Z" or "T" based on the file name and decription
    statisticmap_filtered["map_type"] = statisticmap_filtered.apply(derive_map_type, axis=1)
    statisticmap_filtered = statisticmap_filtered[statisticmap_filtered.map_type.notnull()]

    # Rename columns before merging with collections_pmid_df
    statisticmap_filtered = statisticmap_filtered.rename(
        columns={"image_ptr_id": "image_id", "file": "image_file", "name": "image_name"}
    )
    statisticmap_filtered = statisticmap_filtered[KEEP_IMG_COLUMNS]

    statisticmap_colelctions = pd.merge(
        statisticmap_filtered,
        collections_pmid_df,
        how="left",
        on="collection_id",
    )
    sorted_columns = KEEP_COL_COLUMNS + KEEP_IMG_COLUMNS
    statisticmap_colelctions = statisticmap_colelctions[sorted_columns]

    # Get the cognitive paradigm names
    statisticmap_colelctions = pd.merge(
        statisticmap_colelctions,
        cogat_tasks,
        how="left",
        left_on="cognitive_paradigm_cogatlas_id",
        right_on="cog_atlas_id",
    )
    statisticmap_colelctions = statisticmap_colelctions.rename(
        columns={"name": "cognitive_paradigm_cogatlas_name"}
    )
    statisticmap_colelctions = statisticmap_colelctions.drop(columns=["cog_atlas_id"])

    # Get the cognitive contrast names
    statisticmap_colelctions = pd.merge(
        statisticmap_colelctions,
        cogat_contrasts,
        how="left",
        left_on="cognitive_contrast_cogatlas_id",
        right_on="cog_atlas_id",
    )
    statisticmap_colelctions = statisticmap_colelctions.rename(
        columns={"name": "cognitive_contrast_cogatlas_name"}
    )
    statisticmap_colelctions = statisticmap_colelctions.drop(columns=["cog_atlas_id"])

    # Keep downloaded images only
    image_ids = statisticmap_colelctions["image_id"].unique()
    usable_images_dict = download_images(image_ids, image_dir)
    usable_images_df = pd.DataFrame(usable_images_dict)
    print(f"Usable images: {len(usable_images_df)}/{len(image_ids)}")

    nv_collections_images_df = pd.merge(statisticmap_colelctions, usable_images_df, on="image_id")

    # Add "99999999" to collections with no PMIDs
    nv_collections_images_df["pmid"] = nv_collections_images_df["pmid"].fillna(99999999)

    nv_collections_images_df.to_csv(
        op.join(data_dir, "nv_all_collections_images.csv"), index=False
    )


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
