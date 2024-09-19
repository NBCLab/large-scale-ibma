import argparse
import os
import os.path as op
import warnings

from nimare.dataset import Dataset
from nimare.diagnostics import Jackknife
from nimare.meta.ibma import FixedEffectsHedges
from nimare.workflows import IBMAWorkflow

from ibma import AverageHedges
from utils import _exclude_outliers, _rm_nonstat_maps

warnings.filterwarnings("ignore")

TASK_IDS = {
    "working_memory": [
        "trm_550b50095d4a3",  # working memory fMRI task paradigm
        "tsk_4a57abb949a0d",  # digit span task
        "tsk_4a57abb949bcd",  # n-back task
    ],
}
MANUAL_SELECTION = {
    "working_memory": [
        42805,
        42807,
        42803,
        50291,
        57499,
        58192,
        109901,  # maybe
        109902,
        111344,
        377201,
        405163,
        442126,
        787480,
    ],
}

ESTIMATORS = {
    "mean": AverageHedges(method="mean", aggressive_mask=False),
    "median": AverageHedges(method="median", aggressive_mask=False),
    "trimmed_mean": AverageHedges(method="trimmed", aggressive_mask=False),
    "winsorized_mean": AverageHedges(method="winsorized", aggressive_mask=False),
    "fixed_effects": FixedEffectsHedges(aggressive_mask=False),
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


def main(project_dir, n_cores=1):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    image_dir = op.join(data_dir, "neurovault", "images")
    results_dir = op.join(project_dir, "results", "ibma")
    n_cores = int(n_cores)

    dset = Dataset.load(op.join(data_dir, "neurovault_all_dataset.pkl"))
    dset.update_path(image_dir)
    metadata_df = dset.metadata

    modes = ["all", "heuristic", "manual"]
    tasks = ["working_memory"]  # working memory fMRI task paradigm
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
            if mode == "all":
                metadata_sel_df = metadata_task_df[metadata_task_df["collection_id"] != 457]
                ids_sel = metadata_sel_df["id"].values

            elif mode == "heuristic":
                # Remove images without pmid
                metadata_task_pmid_df = metadata_task_df[metadata_task_df["pmid"] != "99999999"]
                dset_task = dset_task.slice(metadata_task_pmid_df["id"].values)

                # Exclude outliers and non-stat maps
                dset_task = _rm_nonstat_maps(dset_task)
                dset_task = _exclude_outliers(dset_task)
                metadata_sel_df = dset_task.metadata

                # Exclude HCP images
                # This should be remove by our automtic selection anyway
                metadata_sel_df = metadata_sel_df[metadata_sel_df["collection_id"] != 457]
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

            unique_cols = metadata_sel_df["collection_id"].unique()
            n_collections = len(unique_cols)
            print(f"\t{len(ids_sel)} images selected")
            print(f"\t{n_collections} collections")
            for col in unique_cols:
                # Print collection id and images:
                col_df = metadata_sel_df[metadata_sel_df["collection_id"] == col]
                print(
                    f"\t\tCollection {col}, pmid {col_df['pmid'].unique()}: {col_df['image_id'].values}"
                )

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
                diagnostics = Jackknife(voxel_thresh=100, n_cores=n_cores)
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
