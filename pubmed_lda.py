import argparse
import os.path as op

from nimare.annotate.lda import LDAModel
from nimare.dataset import Dataset


from utils import _exclude_outliers


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
        default=4,
        required=False,
        help="CPUs",
    )
    return parser


def main(project_dir, n_cores):
    project_dir = op.abspath(project_dir)
    n_cores = int(n_cores)
    N_TOPICS = 25

    data_dir = op.join(project_dir, "data")
    images_dir = op.join(data_dir, "pubmed_images")
    result_dir = op.join(project_dir, "results", "pubmed_ibma")

    dset_fn = op.join(result_dir, "pubmed_dataset-raw.pkl.gz")
    dset_lda_fn = op.join(result_dir, "pubmed-lda_dataset-raw.pkl.gz")
    dset_clean_fn = op.join(result_dir, "pubmed_dataset-filtered.pkl.gz")
    dset_lda_clean_fn = op.join(result_dir, "pubmed-lda_dataset-filtered.pkl.gz")

    dset = Dataset.load(dset_fn)
    dset.update_path(images_dir)

    if not op.isfile(dset_clean_fn):
        dset_clean = _exclude_outliers(dset)
        dset_clean.save(dset_clean_fn)
    else:
        dset_clean = Dataset.load(dset_clean_fn)

    if not op.isfile(dset_lda_fn):
        model_lda_fn = op.join(result_dir, "pubmed-lda_model-raw.pkl.gz")
        model = LDAModel(n_topics=N_TOPICS, max_iter=1000, text_column="abstract", n_cores=n_cores)
        dset_lda = model.fit(dset)
        dset_lda.save(dset_lda_fn)
        model.save(model_lda_fn)

    if not op.isfile(dset_lda_clean_fn):
        model_lda_clean_fn = op.join(result_dir, "pubmed-lda_model-filtered.pkl.gz")

        model = LDAModel(n_topics=N_TOPICS, max_iter=1000, text_column="abstract", n_cores=n_cores)
        dset_lda_clean = model.fit(dset_clean)
        dset_lda_clean.save(dset_lda_clean_fn)
        model.save(model_lda_clean_fn)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
