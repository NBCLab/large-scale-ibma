import argparse
import os
import os.path as op

import pandas as pd
from nimare.dataset import Dataset
from nimare.meta.ibma import Stouffers
from nimare.workflows import IBMAWorkflow
from nimare.reports.base import run_reports
from nimare.results import MetaResult


def _get_parser():
    parser = argparse.ArgumentParser(description="Run gradient-decoding workflow")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    parser.add_argument(
        "--job_id",
        dest="job_id",
        required=True,
        help="Job ID",
    )
    parser.add_argument(
        "--n_cores",
        dest="n_cores",
        default=4,
        required=False,
        help="CPUs",
    )
    return parser


def main(project_dir, job_id, n_cores):
    n_cores = int(n_cores)
    project_dir = op.abspath(project_dir)

    images_dir = op.join(project_dir, "data", "images")
    result_dir = op.join(project_dir, "results")
    cogat_df = pd.read_csv(op.join(project_dir, "data", "cogat_terms.csv"))

    task_names = cogat_df["cogat_nm"].tolist()
    task_name = task_names[job_id]  # Parallelize using job_id in Slurm

    print(f"Running {task_name}")
    output_dir = op.join(result_dir, "ibma", task_name)
    os.makedirs(output_dir, exist_ok=True)

    z_dset = Dataset.load(op.join(output_dir, f"{task_name}_dset.pkl.gz"))
    z_dset.update_path(images_dir)

    result_fn = op.join(output_dir, f"{task_name}_result-corr.pkl.gz")
    if not op.isfile(result_fn):
        workflow = IBMAWorkflow(estimator=Stouffers(aggressive_mask=False))
        result = workflow.fit(z_dset)
        result.save(result_fn)
        run_reports(result, output_dir)
    else:
        result = MetaResult.load(result_fn)
        run_reports(result, output_dir)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
