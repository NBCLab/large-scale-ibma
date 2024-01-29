import argparse
import os
import os.path as op

import pandas as pd
import numpy as np
import nibabel as nib
from nilearn._utils.niimg_conversions import _check_same_fov
from nilearn.image import concat_imgs, resample_to_img
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


def _get_outliers(data, ids):
    n_studies = data.shape[0]

    mask = ~np.isnan(data) & (data != 0)
    maps_lst = [data[i][mask[i]] for i in range(n_studies)]

    scores = [np.var(map_) for map_ in maps_lst]

    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1

    threshold = 1.5 * iqr
    outliers_idxs = np.where((scores < q1 - threshold) | (scores > q3 + threshold))

    return ids[outliers_idxs]


def _exclude_outliers(dset):
    images = dset.get_images(imtype="z")
    masker = dset.masker

    imgs = [
        nib.load(img)
        if _check_same_fov(nib.load(img), reference_masker=masker.mask_img)
        else resample_to_img(nib.load(img), masker.mask_img)
        for img in images
    ]

    img4d = concat_imgs(imgs, ensure_ndim=4)
    data = masker.transform(img4d)

    outliers = _get_outliers(data, dset.ids)
    unique_ids = np.setdiff1d(dset.ids, outliers)

    return dset.slice(unique_ids)


def main(project_dir, dset_type, job_id, n_cores):
    project_dir = op.abspath(project_dir)
    job_id = int(job_id)
    n_cores = int(n_cores)

    result_dir = op.join(project_dir, "results")
    cogat_df = pd.read_csv(op.join(project_dir, "data", "cogat_terms.csv"))

    task_names = cogat_df["cogat_nm"].tolist()
    task_name = task_names[job_id]  # Parallelize using job_id in Slurm

    print(f"Running task {job_id+1}/{len(task_names)}: {task_name}", flush=True)
    output_dir = op.join(result_dir, "ibma", task_name)
    images_dir = op.join(project_dir, "data", "images", task_name)
    report_dir = op.join(output_dir, f"report-{dset_type}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    z_dset_dict = {}
    z_dset_fn = op.join(output_dir, f"{task_name}_dset-raw.pkl.gz")  # At least this must exist
    z_dset = Dataset.load(z_dset_fn)
    z_dset.update_path(images_dir)
    z_dset_dict["raw"] = z_dset

    if dset_type == "clean":
        z_dset_clean_fn = op.join(output_dir, f"{task_name}_dset-clean.pkl.gz")
        if not op.isfile(z_dset_clean_fn):
            z_dset_clean = _exclude_outliers(z_dset)
            z_dset_clean.save(z_dset_clean_fn)
        else:
            z_dset_clean = Dataset.load(z_dset_clean_fn)

        z_dset_dict["clean"] = z_dset_clean

        print(f"Included Studies: {len(z_dset_clean.ids)}/{len(z_dset.ids)}", flush=True)

    result_fn = op.join(output_dir, f"{task_name}_result-{dset_type}.pkl.gz")
    if not op.isfile(result_fn):
        workflow = IBMAWorkflow(estimator=Stouffers(aggressive_mask=False), n_cores=n_cores)
        result = workflow.fit(z_dset_dict[dset_type])
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
