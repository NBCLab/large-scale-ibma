import argparse
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd
from nimare.dataset import Dataset
from nimare.results import MetaResult
from nimare.stats import pearson

from utils import _vol_to_surf, nifti_to_grayordinate


def _get_parser():
    parser = argparse.ArgumentParser(description="Run IBMA workflow")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    return parser


TASK_TO_HCP = {
    "working_memory": "10_tfMRI_WM_2BK-0BK",
    "emotion_processing": "82_tfMRI_EMOTION_FACES-SHAPES",
    "reward_decision_making": "35_tfMRI_GAMBLING_REWARD-PUNISH",  # 35_tfMRI_GAMBLING_REWARD-PUNISH # 31_tfMRI_GAMBLING_REWARD
    "motor": "42_tfMRI_MOTOR_AVG",
    "language": "65_tfMRI_LANGUAGE_STORY-MATH",
    "social_cognition": "73_tfMRI_SOCIAL_TOM-RANDOM",  # 69_tfMRI_SOCIAL_TOM
}

TARGET = {
    "working_memory": "49_working_memory_back_task.nii.gz",
    "emotion_processing": "109_emotional_process_neural.nii.gz",
    "reward_decision_making": "148_reward_motivation_anticipation.nii.gz",
    "motor": "191_motor_sma_m1.nii.gz",
    "language": "15_language_linguistic_left.nii.gz",
    "social_cognition": "188_social_attachment_social_cognition.nii.gz",
    "response_inhibition": "75_inhibition_response_inhibition_nogo.nii.gz",
    "risk": "131_risk_alcohol_risk_taking.nii.gz",
    "emotion_regulation": "147_regulation_emotion_regulation_reappraisal.nii.gz",
    "visual_perception": "110_modality_visual_sensory.nii.gz",
    "pain": "95_pain_painful_rating.nii.gz",
}

RESULTS_DICT = {
    "mean": {
        "title": "Mean",
        "file_name": "mean_result.pkl.gz",
    },
    "median": {
        "title": "Median",
        "file_name": "median_result.pkl.gz",
    },
    "trimmed_mean": {
        "title": "Trimmed Mean",
        "file_name": "trimmed_mean_result.pkl.gz",
    },
    "winsorized_mean": {
        "title": "Winsorized Mean",
        "file_name": "winsorized_mean_result.pkl.gz",
    },
    "fixed_effects": {
        "title": "Fixed Effects",
        "file_name": "fixed_effects_result.pkl.gz",
    },
}


def _permtest_pearson(data, data_null, target):
    """Permutation test for Pearson correlation."""
    # Calculate true correlations
    corrs = pearson(data, target)

    # Calculate null correlations
    n_perm = len(data_null)
    corrs_null = [pearson(p_i, target) for p_i in data_null]

    # Calculate p-values
    n_extreme_corrs = np.sum(np.abs(corrs_null) >= np.abs(corrs), axis=0)[0]
    return n_extreme_corrs / (n_perm + 1)


def main(project_dir, perm=True):
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results")
    ibma_root_dir = op.join(results_dir, "ibma")
    # ibma_root_dir = op.join(results_dir, "ibma-lda")
    hcp_dir = op.join(data_dir, "hcp")
    hcp_group_dir = op.join(hcp_dir, "HCP_S1200_GroupAvg_v1")
    hcp_tasks_dir = op.join(hcp_dir, "tasks")
    target_path = op.join(data_dir, "neuroquery", "metamaps")

    dset = Dataset.load(op.join(data_dir, "neurovault_all_dataset.pkl"))

    hcp_fn = op.join(
        hcp_group_dir,
        "HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s2_MSMAll.dscalar.nii",
    )
    hcp_maps = nib.load(hcp_fn)
    hcp_data = hcp_maps.get_fdata(dtype=np.float32)
    cifti_hdr = hcp_maps.header
    hcp_axis = cifti_hdr.get_axis(1)
    volume_mask = hcp_axis.volume_mask

    modes = [
        "all",
        # "heuristic",
        # "heuristic-knn",
        # "heuristic-basic",
        "heuristic-advanced",
        # "heuristic-basic+advanced",
        "manual",
    ]
    tasks = [
        # "working_memory",
        # "motor",
        # "pain",
        # "emotion_processing",
        "social_cognition",
        # "response_inhibition",
        # "reward_decision_making",
        # "risk",
        # "visual_perception",
    ]
    for task in tasks:
        task_dir = op.join(ibma_root_dir, task)
        target = (
            "hcp"
            if task
            in [
                "working_memory",
                "motor",
                "emotion_processing",
                "social_cognition",
            ]
            else "cbma"
        )

        if target == "hcp":
            # Get HCP grayordinate for correlation matrix
            target_img_arr = np.load(op.join(hcp_tasks_dir, f"{TASK_TO_HCP[task]}.npy"))[
                ~volume_mask
            ]
        else:
            target_img_fn = op.join(target_path, TARGET[task])
            # target_img = nib.load(target_img_fn)
            # target_img_arr = dset.masker.transform(target_img)
            target_img_arr = _vol_to_surf(target_img_fn)
            target_img_arr = np.nan_to_num(target_img_arr, nan=0)

        # Get top 10% of the target image mask
        target_img_top10_arr = np.sort(target_img_arr)[int(len(target_img_arr) * 0.9) :]
        target_img_top10_mask = np.isin(target_img_arr, target_img_top10_arr)
        target_top_10_fn = op.join(task_dir, "target_top10_data.npy")

        target_top_10_img = np.zeros_like(target_img_arr)
        target_top_10_img[target_img_top10_mask] = target_img_arr[target_img_top10_mask]
        np.save(target_top_10_fn, target_top_10_img)

        corr_dict = {"mode": [], "task": [], "estimator": [], "corr": [], "p_val": []}
        data_dict = {"mode": [], "task": [], "estimator": [], "data": []}
        top_10_dict = {"mode": [], "task": [], "estimator": [], "data": []}

        for mode in modes:
            print(f"Extracting results from task and mode: {task}, {mode}")
            ibma_dir = op.join(task_dir, mode)
            ibma_perm_dir = op.join(ibma_dir, "permutation")

            for col, (label, result_dict) in enumerate(RESULTS_DICT.items()):
                ibma_img_fn = op.join(ibma_dir, f"{label}_map.nii.gz")
                # if not op.isfile(ibma_img_fn):
                result_fn = result_dict["file_name"]
                file_name = op.join(ibma_dir, result_fn)
                result = MetaResult.load(file_name)
                ibma_img = result.get_map("est")
                ibma_img_arr = result.maps["est"]

                nib.save(ibma_img, ibma_img_fn)

                ibma_img_arr = _vol_to_surf(ibma_img_fn)
                ibma_img_arr = np.nan_to_num(ibma_img_arr, nan=0)
                """
                if target == "hcp":
                    # Get image grayordinate for correlation matrix
                    
                    # if label == "trimmed_mean":
                    #    print(ibma_img_arr)
                else:
                    ibma_img_arr = result.maps["est"]
                """

                if (mode.startswith("heuristic") or mode == "manual") and perm:
                    n_perm = 100
                    data_null = []
                    for i in range(n_perm):
                        # Define temp out files
                        ibma_img_perm_fn = op.join(
                            ibma_perm_dir, f"{label}_perm-{i:02d}_map.nii.gz"
                        )

                        # Get image grayordinate for correlation matrix
                        ibma_img_arr_null = _vol_to_surf(ibma_img_perm_fn)

                        # Set nan to zero
                        data_null.append(np.nan_to_num(ibma_img_arr_null, nan=0))

                    p_val = _permtest_pearson(ibma_img_arr, data_null, target_img_arr)
                else:
                    # Calculate p-value only for heuristic-knn
                    p_val = 0

                corr = pearson(ibma_img_arr, target_img_arr)[0]

                ibma_img_top10_arr = ibma_img_arr[target_img_top10_mask]

                corr_dict["mode"].append(mode)
                corr_dict["task"].append(task)
                corr_dict["estimator"].append(label)
                corr_dict["corr"].append(corr)
                corr_dict["p_val"].append(p_val)

                data_dict["data"].extend(ibma_img_arr)
                data_dict["mode"].extend([mode] * len(ibma_img_arr))
                data_dict["task"].extend([task] * len(ibma_img_arr))
                data_dict["estimator"].extend([label] * len(ibma_img_arr))

                top_10_dict["data"].extend(ibma_img_top10_arr)
                top_10_dict["mode"].extend([mode] * len(ibma_img_top10_arr))
                top_10_dict["task"].extend([task] * len(ibma_img_top10_arr))
                top_10_dict["estimator"].extend([label] * len(ibma_img_top10_arr))

                # Save arrays
                data_fn = op.join(ibma_dir, f"{label}_data.npy")
                np.save(data_fn, ibma_img_arr)

                top_10_img = np.zeros_like(ibma_img_arr)
                top_10_img[target_img_top10_mask] = ibma_img_arr[target_img_top10_mask]
                top_10_fn = op.join(ibma_dir, f"{label}_top10_data.npy")
                np.save(top_10_fn, top_10_img)

        corr_df = pd.DataFrame(corr_dict)
        corr_df["mode"] = corr_df["mode"].replace({"heuristic-advanced": "heuristic"})
        corr_df.to_csv(op.join(task_dir, "corr.csv"), index=False)

        data_df = pd.DataFrame(data_dict)
        data_df["mode"] = data_df["mode"].replace({"heuristic-advanced": "heuristic"})
        data_df.to_csv(op.join(task_dir, "data.csv"), index=False)

        top_10_df = pd.DataFrame(top_10_dict)
        top_10_df["mode"] = top_10_df["mode"].replace({"heuristic-advanced": "heuristic"})
        top_10_df.to_csv(op.join(task_dir, "data_top10.csv"), index=False)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
