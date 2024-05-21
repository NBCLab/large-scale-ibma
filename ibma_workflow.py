import argparse
import os
import os.path as op

import pandas as pd
from nimare.dataset import Dataset
from nimare.meta.ibma import Stouffers
from nimare.diagnostics import Jackknife
from nimare.reports.base import run_reports
from nimare.results import MetaResult
from nimare.workflows import IBMAWorkflow

from utils import _rm_nonstat_maps


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
    results_dir = op.join(project_dir, "results")
    stouffers_dir = op.join(results_dir, "stouffers-nv_clean")
    report_dir = op.join(stouffers_dir, "report")
    n_cores = int(n_cores)
    tasks = None
    label = "neurovault"
    os.makedirs(report_dir, exist_ok=True)

    dset = Dataset.load(op.join(results_dir, "neurovault_full_dataset.pkl"))

    nv_collections_images_df = pd.read_csv(op.join(data_dir, "nv_collections_images.csv"))

    if tasks is not None:
        nv_collections_images_df = nv_collections_images_df[
            nv_collections_images_df["cognitive_paradigm_cogatlas_id"].isin(tasks)
        ]

        dset = dset.slice(nv_collections_images_df["id"].values)

    print(dset.metadata)
    print(dset.metadata.columns)
    print(dset.images.shape)

    dset = _rm_nonstat_maps(dset)

    print(dset.images.shape)

    result_fn = op.join(stouffers_dir, f"{label}_result.pkl.gz")
    if not op.isfile(result_fn):
        estimator = Stouffers(use_sample_size=True, aggressive_mask=False)  # aggressive_mask=False
        # Set voxel_thresh to a high value to skip diagnostics for now
        diagnostics = Jackknife(voxel_thresh=100, n_cores=n_cores)
        workflow = IBMAWorkflow(estimator=estimator, diagnostics=diagnostics, n_cores=n_cores)

        result = workflow.fit(dset)
        result.save(result_fn)
    else:
        result = MetaResult.load(result_fn)

    if not op.isfile(op.join(report_dir, "report.html")):
        run_reports(result, report_dir)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
