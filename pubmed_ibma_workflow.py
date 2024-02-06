import argparse
import os
import os.path as op

from nimare.dataset import Dataset
from nimare.meta.ibma import Stouffers
from nimare.workflows import IBMAWorkflow
from nimare.correct import FDRCorrector
from nimare.reports.base import run_reports
from nimare.results import MetaResult
from nimare.meta.ibma import Stouffers
from nimare.workflows import IBMAWorkflow
from nimare.reports.base import run_reports

from utils import _get_studies_to_keep


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
    N_TOPICS = 100
    freq_thr = 0.05
    min_img_thr = 20

    data_dir = op.join(project_dir, "data")
    images_dir = op.join(data_dir, "pubmed_images")
    result_dir = op.join(project_dir, "results", "pubmed_ibma")

    dset_lda_fn = op.join(result_dir, "pubmed-lda_dataset-filtered.pkl.gz")

    dset = Dataset.load(dset_lda_fn)
    dset.update_path(images_dir)

    # Get topics with more than 20 images and at least 5% of the dataset
    feature_group = f"LDA{N_TOPICS}__"
    _, feature_names, feature_ids = _get_studies_to_keep(
        dset,
        feature_group,
        min_img_thr=min_img_thr,
        freq_thr=freq_thr,
    )

    feature = feature_names[job_id]  # Parallelize using job_id in Slurm
    feature_ids = feature_ids[job_id]

    # Create output directories
    ibma_dir = op.join(result_dir, f"nq-lda_{N_TOPICS}", feature)
    report_dir = op.join(ibma_dir, "report")
    os.makedirs(ibma_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print(f"Processing {feature}. {job_id}/{len(feature_names)}", flush=True)
    print(f"{len(feature_ids)}/{len(dset.ids)} studies", flush=True)
    feature_dset = dset.slice(feature_ids)

    result_fn = op.join(ibma_dir, f"{feature}_result.pkl.gz")
    if not op.isfile(result_fn):
        estimator = Stouffers(use_sample_size=True, aggressive_mask=False)
        result = estimator.fit(feature_dset)
        corrector = FDRCorrector()
        corr_result = corrector.transform(result)
        corr_result.save(result_fn)

    #    workflow = IBMAWorkflow(estimator=estimator, n_cores=n_cores)
    #    result = workflow.fit(feature_dset)
    #    result.save(result_fn)
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
