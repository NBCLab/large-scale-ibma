import logging
import math

import numpy as np
from nilearn.input_data import NiftiMasker
from nimare.meta.ibma import IBMAEstimator
from nimare.transforms import d_to_g, t_to_d, z_to_p, z_to_t
from nimare.utils import _boolean_unmask
from scipy.stats import norm

LGR = logging.getLogger(__name__)


def calculate_means(estimates, n_maps, gamma=0.2, method="mean"):
    K = estimates.shape[0]  # K samples, V voxels

    if method == "mean":
        est_maps = estimates.mean(axis=0)
        mean_absolute_deviation = np.abs(estimates - est_maps).mean(axis=0)
        var_maps = mean_absolute_deviation**2

        var_mean_maps = (1 / n_maps[0, :]) * var_maps
        return est_maps, var_mean_maps

    elif method == "median":
        est_maps = np.median(estimates, axis=0)
        median_absolute_deviation = np.median(np.abs(estimates - est_maps), axis=0) * (
            1 / norm.ppf(3 / 4)
        )
        var_maps = median_absolute_deviation**2

        var_mean_maps = (math.pi / (2 * n_maps[0, :])) * var_maps
        return est_maps, var_mean_maps

    elif (method == "trimmed") or (method == "winsorized"):
        K_gamma = int(gamma * K / 2)

        # Sort the estimates along each voxel
        estimates_sorted = np.sort(estimates, axis=0)

        # Trimmed mean calculation
        estimates_trimmed = estimates_sorted[K_gamma:-K_gamma, :]
        trimmed_mean = np.mean(estimates_trimmed, axis=0)

        # Windsorized mean calculation
        estimates_winsorized = estimates_sorted.copy()
        estimates_winsorized[:K_gamma, :] = estimates_sorted[K_gamma, :]
        estimates_winsorized[-K_gamma:, :] = estimates_sorted[-K_gamma - 1, :]
        winsorized_mean = np.mean(estimates_winsorized, axis=0)

        # Windsorized estimate of data variance
        winsorized_var = (
            K_gamma * (estimates_sorted[K_gamma, :] - winsorized_mean) ** 2
            + np.sum((estimates_winsorized[K_gamma:-K_gamma, :] - winsorized_mean) ** 2, axis=0)
            + K_gamma * (estimates_sorted[-K_gamma - 1, :] - winsorized_mean) ** 2
        ) / (K - 1)

        if method == "trimmed":
            # Calculate the variance of the mean
            var_mean_maps = (1 / (n_maps[0, :] - 2 * K_gamma)) * winsorized_var
            return trimmed_mean, var_mean_maps

        elif method == "winsorized":
            # Calculate the variance of the mean
            var_mean_maps = (1 / n_maps[0, :]) * winsorized_var
            return winsorized_mean, var_mean_maps

    else:
        raise ValueError(f"Method {method} not recognized.")


class AverageHedges(IBMAEstimator):
    _required_inputs = {"t_maps": ("image", "t"), "sample_sizes": ("metadata", "sample_sizes")}

    def __init__(self, gamma=0.2, method="mean", **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.method = method

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE"
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} t-statistic images using Heges' g as point estimates "
            "and the variance of bias-corrected Cohen's in a Weighted Least Squares approach "
            "\\citep{brockwell2001comparison,bossier2019}, "
            f"with an a priori tau-squared value of defined across all voxels."
        )
        return description

    def _fit_model(self, t_maps, study_mask=None):
        """Fit the model to the data."""
        n_studies, n_voxels = t_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_studies)

        sample_sizes = np.array([np.mean(self.inputs_["sample_sizes"][idx]) for idx in study_mask])
        n_maps = np.tile(sample_sizes, (n_voxels, 1)).T

        # Calculate Hedge's g maps: Standardized mean
        cohens_maps = t_to_d(t_maps, n_maps)
        hedges_maps = d_to_g(cohens_maps, n_maps)

        del sample_sizes, cohens_maps

        est_maps, var_mean_maps = calculate_means(
            hedges_maps,
            n_maps,
            gamma=self.gamma,
            method=self.method,
        )

        z_map = est_maps / np.sqrt(var_mean_maps)
        p_map = z_to_p(z_map)
        dof = est_maps.shape[0] - 1

        t_map = z_to_t(z_map, dof)
        cohens_maps = t_to_d(t_map, dof)
        hedges_maps = d_to_g(cohens_maps, dof)

        return z_map, p_map, est_maps, hedges_maps

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(self.inputs_["t_maps"][:, voxel_mask])

            z_map, p_map, est_map, es_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["t_maps"].shape[1]

            z_map, p_map, est_map, es_map = [np.zeros(n_voxels, dtype=float) for _ in range(4)]
            for bag in self.inputs_["data_bags"]["t_maps"]:
                (
                    z_map[bag["voxel_mask"]],
                    p_map[bag["voxel_mask"]],
                    est_map[bag["voxel_mask"]],
                    es_map[bag["voxel_mask"]],
                ) = self._fit_model(bag["values"], bag["study_mask"])

        maps = {"z": z_map, "p": p_map, "est": est_map, "es": es_map}
        description = self._generate_description()

        return maps, {}, description
