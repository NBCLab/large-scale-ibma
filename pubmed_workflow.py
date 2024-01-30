import argparse
import os
import os.path as op

from nimare.dataset import Dataset
from nimare.meta.ibma import Stouffers
from nimare.workflows import IBMAWorkflow
from nimare.reports.base import run_reports
from nimare.results import MetaResult
from nimare.meta.ibma import Stouffers
from nimare.workflows import IBMAWorkflow
from nimare.reports.base import run_reports


def _get_parser():
    parser = argparse.ArgumentParser(description="Run IBMA workflow")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    parser.add_argument(
        "--dset_type",
        dest="dset_type",
        required=True,
        help="Kind of dataset to use",
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


def main(project_dir, dset_type, job_id, n_cores):
    project_dir = op.abspath(project_dir)
    job_id = int(job_id)
    n_cores = int(n_cores)
    N_TOPICS = 25
    frequency_threshold = 0.05

    data_dir = op.join(project_dir, "data")
    images_dir = op.join(data_dir, "pubmed_images")
    result_dir = op.join(project_dir, "results", "pubmed_ibma")

    dset_lda_fn = op.join(result_dir, f"pubmed-lda_dataset-{dset_type}.pkl.gz")

    dset = Dataset.load(dset_lda_fn)
    dset.update_path(images_dir)

    feature_group = f"LDA{N_TOPICS}__"
    feature_names = dset.annotations.columns.values
    feature_names = [f for f in feature_names if f.startswith(feature_group)]
    feature = feature_names[job_id]  # Parallelize using job_id in Slurm

    print(f"Processing {feature}", flush=True)
    ibma_dir = op.join(result_dir, dset_type, feature)
    report_dir = op.join(ibma_dir, "report")
    os.makedirs(ibma_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    feature_ids = dset.get_studies_by_label(
        labels=[feature],
        label_threshold=frequency_threshold,
    )
    feature_dset = dset.slice(feature_ids)
    print(f"{len(feature_ids)}/{len(dset.ids)} studies", flush=True)

    result_fn = op.join(ibma_dir, f"{feature}_result.pkl.gz")
    if not op.isfile(result_fn):
        workflow = IBMAWorkflow(estimator=Stouffers(aggressive_mask=False), n_cores=n_cores)
        result = workflow.fit(feature_dset)
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
