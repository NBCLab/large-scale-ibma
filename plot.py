"""Utility functions for figures."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from neuromaps import transforms
from neuromaps.datasets import fetch_fslr
from nilearn import datasets
from nilearn.plotting import plot_stat_map
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from surfplot import Plot

from utils import _zero_medial_wall

CMAP = nilearn_cmaps["cold_hot"]


def plot_vol(
    nii_img_thr, threshold, out_file, mask_contours=None, coords=None, vmax=8, alpha=1, cmap=CMAP
):
    template = datasets.load_mni152_template(resolution=1)

    display_modes = ["x", "y", "z"]
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    gs = GridSpec(2, 2, figure=fig)

    for dsp_i, display_mode in enumerate(display_modes):
        if display_mode == "z":
            ax = fig.add_subplot(gs[:, 1], aspect="equal")
            colorbar = True
        else:
            ax = fig.add_subplot(gs[dsp_i, 0], aspect="equal")
            colorbar = False

        if coords is not None:
            cut_coords = [coords[dsp_i]]
            if np.isnan(cut_coords):
                cut_coords = 1
        else:
            cut_coords = 1

        display = plot_stat_map(
            nii_img_thr,
            bg_img=template,
            black_bg=False,
            draw_cross=False,
            annotate=True,
            alpha=alpha,
            cmap=cmap,
            threshold=threshold,
            symmetric_cbar=True,
            colorbar=colorbar,
            display_mode=display_mode,
            cut_coords=cut_coords,
            vmax=vmax,
            axes=ax,
        )
        if mask_contours:
            display.add_contours(mask_contours, levels=[0.5], colors="black")

    fig.savefig(out_file, bbox_inches="tight", dpi=300)


def plot_surf(map_lh, map_rh, out_file, mask_contours=None, vmax=8, cmap=CMAP):
    surfaces = fetch_fslr(density="32k")
    lh, rh = surfaces["inflated"]
    sulc_lh, sulc_rh = surfaces["sulc"]

    p = Plot(surf_lh=lh, surf_rh=rh, layout="grid")
    p.add_layer({"left": sulc_lh, "right": sulc_rh}, cmap="binary_r", cbar=False)
    p.add_layer(
        {"left": map_lh, "right": map_rh}, cmap=cmap, cbar=False, color_range=(-vmax, vmax)
    )
    if mask_contours:
        mask_lh, mask_rh = transforms.mni152_to_fslr(mask_contours, fslr_density="32k")
        mask_lh, mask_rh = _zero_medial_wall(
            mask_lh,
            mask_rh,
            space="fsLR",
            density="32k",
        )
        mask_arr_lh = mask_lh.agg_data()
        mask_arr_rh = mask_rh.agg_data()
        countours_lh = np.zeros_like(mask_arr_lh)
        countours_lh[mask_arr_lh != 0] = 1
        countours_rh = np.zeros_like(mask_arr_rh)
        countours_rh[mask_arr_rh != 0] = 1

        colors = [(0, 0, 0, 0)]
        contour_cmap = ListedColormap(colors, "regions", N=1)
        line_cmap = ListedColormap(["black"], "regions", N=1)
        p.add_layer(
            {"left": countours_lh, "right": countours_rh},
            cmap=line_cmap,
            as_outline=True,
            cbar=False,
        )
        p.add_layer(
            {"left": countours_lh, "right": countours_rh},
            cmap=contour_cmap,
            cbar=False,
        )
    fig = p.build()
    fig.savefig(out_file, bbox_inches="tight", transparent=True, dpi=300)
