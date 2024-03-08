"""Utility functions for figures."""

import gc
import math
import os.path as op

import pandas as pd
from cognitiveatlas.api import get_concept
from nimare.utils import get_resource_path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from neuromaps import transforms
from nilearn.plotting import plot_stat_map
from nilearn import datasets
from gradec.utils import _zero_medial_wall
from matplotlib.gridspec import GridSpec
import numpy as np
from neuromaps.datasets import fetch_fslr
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from surfplot import Plot

CMAP = nilearn_cmaps["cold_hot"]


def trim_image(img=None, tol=1, fix=True):
    mask = img != tol if fix else img <= tol
    if img.ndim == 3:
        mask = mask.any(2)
    mask0, mask1 = mask.any(0), mask.any(1)
    mask1[0] = False
    mask1[-1] = False
    return img[:, mask0]


def plot_vol(nii_img_thr, threshold, out_file, mask_contours=None, vmax=8, alpha=1, cmap=CMAP):
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
            cut_coords=1,
            vmax=vmax,
            axes=ax,
        )
        if mask_contours:
            display.add_contours(mask_contours, levels=[0.5], colors="black")

    fig.savefig(out_file, bbox_inches="tight", dpi=300)

    fig = None
    plt.close()
    gc.collect()
    plt.clf()


def plot_surf(nii_img_thr, out_file, mask_contours=None, vmax=8, cmap=CMAP):
    map_lh, map_rh = transforms.mni152_to_fslr(nii_img_thr, fslr_density="32k")
    map_lh, map_rh = _zero_medial_wall(
        map_lh,
        map_rh,
        space="fsLR",
        density="32k",
    )
    # midthickness

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
    fig.savefig(out_file, bbox_inches="tight", dpi=300)

    fig = None
    plt.close()
    gc.collect()
    plt.clf()


def plot_top_words(topic_word_weight, features_name, n_top_words, dpi, out_filename):
    top_features_ind = topic_word_weight.argsort()[: -n_top_words - 1 : -1]
    top_features = [features_name[i] for i in top_features_ind]
    weights = topic_word_weight[top_features_ind]

    fig, ax = plt.subplots(figsize=(7, 9))

    norm = plt.Normalize(0, np.max(weights))
    color = plt.colormaps["YlOrRd"]
    colors = [color(norm(weight)) for weight in weights]

    ax.barh(top_features, weights, height=0.7, color=colors)
    ax.set_title("Topic-word weight", fontdict={"fontsize": 25})
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=20)

    fig.subplots_adjust(left=0.35, right=0.98, top=0.95, bottom=0.05)
    fig.savefig(out_filename, dpi=dpi)
    fig = None
    plt.close()
    gc.collect()
    plt.clf()


def plot_topic_model(model, feature_names, n_top_words, title):
    n_topics = len(model.components_)
    n_cols = 5
    n_rows = math.ceil(n_topics / n_cols)
    w = 30
    h = (w / 2) * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w, h), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.977, bottom=0.05, wspace=0.90, hspace=0.1)
    plt.show()


def _lowercase(words):
    return [word.lower() for word in words]


def _extract_vocabulary(phrase):
    words = phrase.split()
    sub_phrases = [
        _lowercase(words[i:j]) for i in range(len(words)) for j in range(i + 1, len(words) + 1)
    ]
    return [" ".join(sub_phrase) for sub_phrase in sub_phrases]


def _cogat_vocabulary(cogat_dir):
    """Get vocabulary from cognitive atlas concepts."""
    cogat_fn = op.join(cogat_dir, "cogat_concepts.csv")
    if not op.isfile(cogat_fn):
        concepts_df = get_concept(silent=True).pandas
    else:
        concepts_df = pd.read_csv(cogat_fn)

    cog_names = concepts_df["name"].to_list()
    vocabulary = []
    for name in cog_names:
        vocabulary.extend(_extract_vocabulary(name))

    return list(set(vocabulary))


def _generate_counts(
    text_df,
    id_col="id",
    vocabulary=None,
    text_column="abstract",
    tfidf=True,
    min_df=0.01,
    max_df=0.99,
):
    """Generate tf-idf/counts weights for unigrams/bigrams derived from textual data.

    Parameters
    ----------
    text_df : (D x 2) :obj:`pandas.DataFrame`
        A DataFrame with two columns ('id' and 'text'). D = document.

    Returns
    -------
    weights_df : (D x T) :obj:`pandas.DataFrame`
        A DataFrame where the index is 'id' and the columns are the
        unigrams/bigrams derived from the data. D = document. T = term.
    """
    if text_column not in text_df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    # Remove rows with empty text cells
    orig_ids = text_df[id_col].tolist()
    text_df = text_df.fillna("")
    keep_ids = text_df.loc[text_df[text_column] != "", id_col]
    text_df = text_df.loc[text_df[id_col].isin(keep_ids)]

    if len(keep_ids) != len(orig_ids):
        print(f"\t\tRetaining {len(keep_ids)}/{len(orig_ids)} studies", flush=True)

    ids = text_df[id_col].tolist()
    text = text_df[text_column].tolist()
    stoplist = op.join(get_resource_path(), "neurosynth_stoplist.txt")
    with open(stoplist, "r") as fo:
        stop_words = fo.read().splitlines()

    if tfidf:
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            vocabulary=vocabulary,
            stop_words=stop_words,
        )
    else:
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            vocabulary=vocabulary,
            stop_words=stop_words,
        )
    weights = vectorizer.fit_transform(text).toarray()

    names = vectorizer.get_feature_names_out()
    names = [str(name) for name in names]
    weights_df = pd.DataFrame(weights, columns=names, index=ids)
    weights_df.index.name = id_col
    return weights_df
