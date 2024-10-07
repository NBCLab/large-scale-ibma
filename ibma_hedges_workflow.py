import argparse
import logging
import os
import os.path as op
import warnings

import nibabel as nib
from nimare.dataset import Dataset
from nimare.diagnostics import Jackknife
from nimare.meta.ibma import FixedEffectsHedges
from nimare.workflows import IBMAWorkflow

from ibma import AverageHedges
from outlier import remove_outliers

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

TASK_IDS = {
    "working_memory": [
        "trm_550b50095d4a3",  # working memory fMRI task paradigm
        "tsk_4a57abb949a0d",  # digit span task
        "tsk_4a57abb949bcd",  # n-back task
    ],
    "emotion_processing": ["trm_550b5b066d37b"],  # emotion processing fMRI task paradigm
    "reward_decision_making": [
        "trm_4f24496a80587",  # gambling task
        "trm_550b5c1a7f4db",  # gambling fMRI task paradigm
        "trm_4cacee4a1d875",  # mixed gambles task
        "tsk_Ncknr0soiM4IV",  # social decision-making task
    ],
    "motor": [
        "trm_550b53d7dd674",  # motor fMRI task paradigm
        "tsk_4a57abb949bbf",  # motor sequencing task
        "trm_4c898f079d05e",  # finger tapping task
    ],
    "language": ["trm_550b54a8b30f4"],  # language processing fMRI task paradigm
}
MANUAL_SELECTION = {
    "working_memory": [
        42805,
        42807,
        42803,
        # 50291,
        57499,
        58192,
        109901,
        109902,
        111344,
        # 377201,
        405163,
        442126,
        787480,
    ],
    "emotion_processing": [100438, 108833, 564310],
    "reward_decision_making": [
        16208,
        550282,
    ],
    "motor": [],
    "language": [],
}
# EXCLUDE_COLLECTIONS = [457]
# EXCLUDE_COLLECTIONS = [457, 2621]
EXCLUDE_COLLECTIONS = [457, 2621, 7103, 7104]
ESTIMATORS = {
    "mean": AverageHedges(method="mean", aggressive_mask=False),
    "median": AverageHedges(method="median", aggressive_mask=False),
    "trimmed_mean": AverageHedges(method="trimmed", aggressive_mask=False),
    "winsorized_mean": AverageHedges(method="winsorized", aggressive_mask=False),
    "fixed_effects": FixedEffectsHedges(aggressive_mask=False),
}
TARGET_IMGs = {
    "working_memory": nib.load(
        "/Users/julioaperaza/Documents/GitHub/large-scale-ibma/data/neurosynth/metamaps/working_memory.nii.gz"
    ),
    "reward_decision_making": nib.load(
        "/Users/julioaperaza/Documents/GitHub/large-scale-ibma/data/neurosynth/metamaps/reward.nii.gz"
    ),
    "motor": nib.load(
        "/Users/julioaperaza/Documents/GitHub/large-scale-ibma/data/neurosynth/metamaps/motor.nii.gz"
    ),
}


def _verbose_print(metadata_sel_df, ids_sel):
    unique_cols = metadata_sel_df["collection_id"].unique()
    n_collections = len(unique_cols)
    print(f"\t{len(ids_sel)} images selected")
    print(f"\t{n_collections} collections")
    for col in unique_cols:
        # Print collection id and images:
        col_df = metadata_sel_df[metadata_sel_df["collection_id"] == col]
        pmid_link = "https://pubmed.ncbi.nlm.nih.gov/"
        coll_link = "https://neurovault.org/collections/"
        image_link = "https://neurovault.org/images/"

        print(col)
        print(f"Link: {coll_link}{col}")
        pmid_ = col_df["pmid"].unique()[0]
        if pmid_ != "99999999":
            print(f"Paper: {pmid_link}{pmid_}")
        else:
            print("Paper: None")
        print("Images:")
        for img_id in col_df["image_id"].values:
            print(f"{image_link}{img_id}")
        print("")


OUTLIER_METHODS = {
    "heuristic": "None",
    "heuristic-knn": "knn",
    "heuristic-basic": "basic",
    "heuristic-advanced": "advanced",
    "heuristic-basic+advanced": "full",
}


def _get_parser():
    parser = argparse.ArgumentParser(description="Run IBMA workflow")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    parser.add_argument(
        "--n_cores",
        dest="n_cores",
        default=1,
        required=False,
        help="CPUs",
    )
    return parser


def main(project_dir, verbose=0, n_cores=1):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    image_dir = op.join(data_dir, "neurovault", "images")
    results_dir = op.join(project_dir, "results", "ibma")
    n_cores = int(n_cores)

    dset = Dataset.load(op.join(data_dir, "neurovault_all_dataset.pkl"))
    dset.update_path(image_dir)
    metadata_df = dset.metadata

    modes = [
        "all",
        # "heuristic",
        "heuristic-knn",
        # "heuristic-basic",
        # "heuristic-advanced",
        # "heuristic-basic+advanced",
        "manual",
    ]
    # modes = ["heuristic-basic"]
    # tasks = ["working_memory", "reward_decision_making", "motor"]
    # tasks = ["reward_decision_making", "motor"]
    tasks = ["working_memory"]
    for mode in modes:
        print(f"Running IBMA for mode: {mode}")
        for task in tasks:
            print("Running IBMA for task: ", task)

            task_ids = TASK_IDS[task]
            sub_metadata_df = metadata_df[
                metadata_df["cognitive_paradigm_cogatlas_id"].isin(task_ids)
            ]
            task_name = sub_metadata_df["cognitive_paradigm_cogatlas_name"].unique()

            print(f"\tTask: {task_name}")
            dset_task = dset.slice(sub_metadata_df["id"].values)
            metadata_task_df = dset_task.metadata

            # Remove HCP images
            non_hcp_df = metadata_task_df[
                ~metadata_task_df["collection_id"].isin(EXCLUDE_COLLECTIONS)
            ]
            dset_task = dset_task.slice(non_hcp_df["id"].values)

            if mode == "all":
                metadata_sel_df = dset_task.metadata
                ids_sel = metadata_sel_df["id"].values

            elif mode.startswith("heuristic"):
                # Remove images without pmid
                metadata_task_pmid_df = metadata_task_df[metadata_task_df["pmid"] != "99999999"]
                dset_task = dset_task.slice(metadata_task_pmid_df["id"].values)

                # Exclude outliers and non-stat maps
                dset_task = remove_outliers(
                    dset_task,
                    method=OUTLIER_METHODS[mode],
                    target=TARGET_IMGs[task],
                )
                metadata_sel_df = dset_task.metadata
                ids_sel = metadata_sel_df["id"].values

                if len(ids_sel) < 2:
                    print(f"Skipping task {task} because it has no more than 2 images")
                    continue

            elif mode == "manual":
                img_ids = MANUAL_SELECTION[task]
                metadata_sel_df = metadata_task_df[metadata_task_df["image_id"].isin(img_ids)]
                ids_sel = metadata_sel_df["id"].values

            else:
                raise ValueError("Invalid mode")

            # if verbose > 0:
            if mode == "heuristic-basic":
                _verbose_print(metadata_sel_df, ids_sel)

            # Define output directories
            ibma_dir = op.join(results_dir, task, mode)
            os.makedirs(ibma_dir, exist_ok=True)

            metadata_sel_df.to_csv(op.join(ibma_dir, f"metadata_{mode}.csv"), index=False)

            # Run IBMA on non-HCP
            dset_task_sel = dset.slice(ids_sel)
            print(f"\tRunning IBMA, with {len(dset_task_sel.images)} images")
            for label, estimator in ESTIMATORS.items():
                print("\t\tUsing estimator: ", label)
                result_fn = op.join(ibma_dir, f"{label}_result.pkl.gz")

                # Set voxel_thresh to a high value to skip diagnostics for now
                diagnostics = Jackknife(voxel_thresh=10000, n_cores=n_cores)
                workflow = IBMAWorkflow(
                    estimator=estimator, diagnostics=diagnostics, n_cores=n_cores
                )
                result = workflow.fit(dset_task_sel)
                result.save(result_fn)
                print("\t\t\tDone!")


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
