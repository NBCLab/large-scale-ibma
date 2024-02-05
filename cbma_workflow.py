import argparse
import os
import os.path as op

from nimare.dataset import Dataset
from nimare.meta import ALE
from nimare.correct import FDRCorrector
from nimare.workflows import CBMAWorkflow
from nimare.reports.base import run_reports
from nimare.results import MetaResult


def _get_parser():
    parser = argparse.ArgumentParser(description="Run IBMA workflow")
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
    project_dir = op.abspath(project_dir)
    job_id = int(job_id)
    n_cores = int(n_cores)
    N_TOPICS = 200
    frequency_threshold = 0.05

    data_dir = op.join(project_dir, "data")
    images_dir = op.join(data_dir, "pubmed_images")
    result_dir = op.join(project_dir, "results", "cbma")
    ibma_dir = op.join(project_dir, "results", "pubmed_ibma")

    dset_nq_lda_fn = op.join(data_dir, "neuroquery", "neuroquery_lda_dataset.pkl.gz")
    dset_lda_fn = op.join(ibma_dir, "pubmed-lda_dataset-filtered.pkl.gz")

    nq_dset = Dataset.load(dset_nq_lda_fn)
    dset = Dataset.load(dset_lda_fn)
    dset.update_path(images_dir)

    feature_group = f"LDA{N_TOPICS}__"
    feature_names = dset.annotations.columns.values
    feature_names = [f for f in feature_names if f.startswith(feature_group)]

    # Get topics with more than 10 images and at least 5% of the dataset
    lda_feature_names_keep = []
    for feature in feature_names:
        temp_feature_ids = dset.get_studies_by_label(
            labels=[feature],
            label_threshold=frequency_threshold,
        )
        if len(temp_feature_ids) >= 10:
            lda_feature_names_keep.append(feature)

    feature = lda_feature_names_keep[job_id]  # Parallelize using job_id in Slurm
    feature_ids = nq_dset.get_studies_by_label(
        labels=[feature],
        label_threshold=frequency_threshold,
    )

    # Create output directories
    cbma_dir = op.join(result_dir, "nq-lda_200", feature)
    report_dir = op.join(cbma_dir, "report")
    os.makedirs(cbma_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print(f"Processing {feature}. {job_id}/{len(lda_feature_names_keep)}", flush=True)
    print(f"{len(feature_ids)}/{len(nq_dset.ids)} studies", flush=True)
    feature_dset = nq_dset.slice(feature_ids)

    result_fn = op.join(cbma_dir, f"{feature}_result.pkl.gz")
    if not op.isfile(result_fn):
        estimator = ALE(kernel__sample_size=20)
        result = estimator.fit(feature_dset)
        corrector = FDRCorrector()
        corr_result = corrector.transform(result)
        corr_result.save(result_fn)

    # Skip the Workflow for now. It is time consuming and not necessary for the current task
    """
    if not op.isfile(result_fn):
        workflow = CBMAWorkflow(
            estimator=ALE(kernel__sample_size=20),
            corrector="fdr",
            diagnostics=["jackknife"],
            n_cores=n_cores,
        )
        result = workflow.fit(feature_dset)
        result.save(result_fn)
    else:
        result = MetaResult.load(result_fn)

    if not op.isfile(op.join(report_dir, "report.html")):
        run_reports(result, report_dir)
    """


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
