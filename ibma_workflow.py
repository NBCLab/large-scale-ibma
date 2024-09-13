import argparse
import os
import os.path as op
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from nilearn import datasets
from nilearn.plotting import plot_stat_map
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from nimare.dataset import Dataset
from nimare.diagnostics import Jackknife
from nimare.meta.ibma import FixedEffectsHedges, Stouffers
from nimare.reports.base import run_reports
from nimare.results import MetaResult
from nimare.transforms import ImageTransformer
from nimare.workflows import IBMAWorkflow

from utils import _exclude_outliers, _rm_nonstat_maps

CMAP = nilearn_cmaps["cold_hot"]

warnings.filterwarnings("ignore")

MANUAL_SELECTION = {
    "trm_550b50095d4a3": [
        59429,
        57498,
        58192,
        111340,
        377201,
        405163,
    ],  # working memory fMRI task paradigm
}


def plot_vol(
    nii_img_thr,
    threshold,
    out_file,
    mask_contours=None,
    coords=None,
    vmax=8,
    alpha=1,
    title=None,
    cmap=CMAP,
):
    template = datasets.load_mni152_template(resolution=1)

    display_modes = ["x", "y", "z"]
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    gs = GridSpec(2, 2, figure=fig)

    for dsp_i, display_mode in enumerate(display_modes):
        if display_mode == "z":
            ax = fig.add_subplot(gs[:, 1], aspect="equal")
            colorbar = True
        else:
            ax = fig.add_subplot(gs[dsp_i, 0], aspect="equal")
            colorbar = False

        if coords is not None:
            cut_coords = [coords[dsp_i]]
            if np.isnan(cut_coords):
                cut_coords = 1
        else:
            cut_coords = 1

        display = plot_stat_map(
            nii_img_thr,
            bg_img=template,
            black_bg=False,
            draw_cross=False,
            annotate=True,
            alpha=alpha,
            cmap=cmap,
            threshold=threshold,
            symmetric_cbar=True,
            colorbar=colorbar,
            display_mode=display_mode,
            cut_coords=cut_coords,
            vmax=vmax,
            axes=ax,
        )
        if mask_contours:
            display.add_contours(mask_contours, levels=[0.5], colors="black")

    if title is not None:
        fig.suptitle(title, fontsize=16)

    fig.savefig(out_file, bbox_inches="tight", dpi=300)


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
    image_dir = op.join(data_dir, "nv-data", "images")
    results_dir = op.join(project_dir, "results", "ibma")
    n_cores = int(n_cores)

    estimators = [
        Stouffers(use_sample_size=False, normalize_contrast_weights=False, aggressive_mask=False),
        Stouffers(use_sample_size=True, normalize_contrast_weights=False, aggressive_mask=False),
        Stouffers(use_sample_size=False, normalize_contrast_weights=False, aggressive_mask=False),
        Stouffers(use_sample_size=True, normalize_contrast_weights=False, aggressive_mask=False),
        Stouffers(use_sample_size=True, normalize_contrast_weights=True, aggressive_mask=False),
        FixedEffectsHedges(aggressive_mask=False),
    ]
    estimators_labels = [
        "stouffers",
        "stouffers-weighted",
        "stouffers-ncontrast",
        "stouffers-ncontrast-weighted",
        "stouffers-ncontrast-weighted-downweighted",
        "fe-hedges",
    ]

    dset = Dataset.load(op.join(data_dir, "neurovault_full_dataset.pkl"))
    dset.update_path(image_dir)

    # tasks = ["trm_4f2453ce33f16"]  # Test with social judgment task
    # tasks = ["trm_550b5b066d37b"]  # emotion processing fMRI task paradigm
    tasks = ["trm_550b50095d4a3"]  # working memory fMRI task paradigm
    modes = ["manual", "auto"]
    # Get the tasks for the collection_id 457 (HCP)
    metadata_df = dset.metadata
    # tasks = metadata_df[metadata_df["collection_id"] == 457][
    #    "cognitive_paradigm_cogatlas_id"
    # ].unique()
    for mode in modes:
        print(f"Running IBMA for mode: {mode}")
        for task in tasks:
            print("Running IBMA for task: ", task)
            sub_metadata_df = metadata_df[metadata_df["cognitive_paradigm_cogatlas_id"] == task]
            task_name = sub_metadata_df["cognitive_paradigm_cogatlas_name"].unique()[0]
            print(f"\tTask: {task_name}")

            task_label = task_name.replace(" ", "_").lower()
            if mode == "auto":
                dset_task = dset.slice(sub_metadata_df["id"].values)

                # Exclude outliers and non-stat maps
                dset_task = _rm_nonstat_maps(dset_task)
                dset_task = _exclude_outliers(dset_task)
                metadata_task_df = dset_task.metadata

                # Split the data in HCP and no HCP
                # This should be remove by our automtic selection anyway
                ids_sel = metadata_task_df[metadata_task_df["collection_id"] != 457]["id"].values

                no_hcp_df = metadata_task_df[metadata_task_df["collection_id"] != 457]
                print(f"\t{len(ids_sel)} no-HCP images")
                n_collections = len(no_hcp_df["collection_id"].unique())
                print(f"\t{n_collections} collections")

                # unique_collections = no_hcp_df["collection_id"].unique()
                # unique_images = no_hcp_df["image_id"].unique()

                if len(ids_sel) < 2:
                    print(f"Skipping task {task_label} because it has no more than 2 images")
                    continue

            elif mode == "manual":
                img_ids = MANUAL_SELECTION[task]
                ids_sel = sub_metadata_df[sub_metadata_df["image_id"].isin(img_ids)]["id"].values
                print(sub_metadata_df["image_id"])
                print(f"\t{len(ids_sel)} images selected manually")
            else:
                raise ValueError("Invalid mode")

            dset_task_sel = dset.slice(ids_sel)

            dset_task_sel_contrast = dset_task_sel.copy()
            dset_task_sel_contrast.images["study_id"] = list(
                range(len(dset_task_sel_contrast.images))
            )

            # Define output directories
            ibma_dir = op.join(results_dir, task_label, mode)
            os.makedirs(ibma_dir, exist_ok=True)

            """
            # Plot the images
            for img in dset_no_hcp.images["z"]:
                img_fn = img.split("/")[-1].split("_")[0]
                vol_fn = op.join(ibma_dir, f"{img_fn}_vol.png")
    
                plot_vol(img, 1.0, vol_fn, title=img_fn, vmax=12)
    
            """
            # Run IBMA on non-HCP
            print(f"\tRunning IBMA for no HCP, with {len(dset_task_sel.images)} images")
            for label, estimator in zip(estimators_labels, estimators):
                print("\t\tUsing estimator: ", label)
                result_fn = op.join(ibma_dir, f"no-hcp_{label}_result.pkl.gz")

                if label in ["stouffers", "stouffers-weighted"]:
                    use_dset_no_hcp = dset_task_sel_contrast
                else:
                    use_dset_no_hcp = dset_task_sel

                # Set voxel_thresh to a high value to skip diagnostics for now
                diagnostics = Jackknife(voxel_thresh=100, n_cores=n_cores)
                workflow = IBMAWorkflow(
                    estimator=estimator, diagnostics=diagnostics, n_cores=n_cores
                )
                result = workflow.fit(use_dset_no_hcp)
                result.save(result_fn)
                print("\t\t\tDone!")
                # else:
                #    result = MetaResult.load(result_fn)

                # if not op.isfile(op.join(report_dir, "report.html")):
                #    run_reports(result, report_dir)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
