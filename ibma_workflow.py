import argparse
import os
import os.path as op

from nimare.dataset import Dataset
from nimare.meta.ibma import Stouffers
from nimare.workflows import IBMAWorkflow
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
    return parser


def main(project_dir):
    project_dir = op.abspath(project_dir)
    results_dir = op.join(project_dir, "results", "cogat_test")

    labels = ["full", "clean", "random", "selected"]
    for label in labels:
        report_dir = op.join(results_dir, f"report-{label}")
        os.makedirs(report_dir, exist_ok=True)

        dset = Dataset.load(op.join(results_dir, f"go-no-go-task_{label}_dataset.pkl"))

        result_fn = op.join(results_dir, f"go-no-go-task_{label}_result.pkl.gz")
        if not op.isfile(result_fn):
            estimator = Stouffers(use_sample_size=True, aggressive_mask=False)
            workflow = IBMAWorkflow(estimator=estimator)

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
