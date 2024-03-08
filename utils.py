"""Utility functions for the meta-analysis pipeline."""

import requests
import os.path as op

import pandas as pd
import numpy as np
from cognitiveatlas.api import get_concept
import nibabel as nib
from nilearn._utils.niimg_conversions import _check_same_fov
from nilearn.image import concat_imgs, resample_to_img
from nimare.utils import get_resource_path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nimare.extract import fetch_neuroquery
from nimare.io import convert_neurosynth_to_dataset


def _get_outliers(data, ids):
    # Here n_voxels is data.shape[1] considering the nonaggresive mask is used
    n_studies, n_voxels = data.shape

    mask = ~np.isnan(data) & (data != 0)
    maps_lst = [data[i][mask[i]] for i in range(n_studies)]

    # Get outliers based on variance within each study
    # Set var of empty maps to 0, it will be remove by perc_outliers
    scores = [np.var(map_) if len(map_) > 1 else 0 for map_ in maps_lst]
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    threshold = 1.5 * iqr
    var_outliers = np.where((scores < q1 - threshold) | (scores > q3 + threshold))[0]

    # Get images with all positive or all negative values
    oth_outliers = np.array(
        [
            map_i
            for map_i, map_ in enumerate(maps_lst)
            if ((map_ > 0).sum() == len(map_)) or ((map_ < 0).sum() == len(map_))
        ],
        dtype=int,
    )

    # Get images with less than 40% of voxels
    perc_voxs = mask.sum(axis=1) / n_voxels
    perc_outliers = np.where(perc_voxs < 0.4)[0]

    outliers_idxs = np.unique(np.hstack([var_outliers, oth_outliers, perc_outliers]))

    return ids[outliers_idxs]


def _exclude_outliers(dset):
    # defaults for resampling images (nilearn's defaults do not work well)
    _resample_kwargs = {"clip": True, "interpolation": "linear"}

    images = dset.get_images(imtype="z")
    masker = dset.masker

    imgs = [
        (
            nib.load(img)
            if _check_same_fov(nib.load(img), reference_masker=masker.mask_img)
            else resample_to_img(nib.load(img), masker.mask_img, **_resample_kwargs)
        )
        for img in images
    ]

    img4d = concat_imgs(imgs, ensure_ndim=4)
    data = masker.transform(img4d)

    outliers = _get_outliers(data, dset.ids)
    unique_ids = np.setdiff1d(dset.ids, outliers)

    return dset.slice(unique_ids)


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


def _fetch_neuroquery_dset():
    files = fetch_neuroquery(
        data_dir="./data",
        version="1",
        overwrite=False,
        source="combined",
        vocab="neuroquery6308",
        type="tfidf",
    )
    neuroquery_db = files[0]

    return convert_neurosynth_to_dataset(
        coordinates_file=neuroquery_db["coordinates"],
        metadata_file=neuroquery_db["metadata"],
        annotations_files=neuroquery_db["features"],
    )


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


def _add_texts(dset, texts_fn):
    """Add texts to NeuroQuery dataset."""
    texts_df = pd.read_csv(texts_fn)
    study_ids = [f"{id_}-1" for id_ in texts_df["pmid"]]
    texts_df["id"] = study_ids
    texts_df = texts_df[["id", "title", "keywords", "abstract", "body"]]
    texts_df.set_index("id", inplace=True)

    new_texts_df = dset.texts.copy()
    new_texts_df = pd.merge(new_texts_df, texts_df, left_on="id", right_index=True)

    new_dset = dset.copy()
    new_dset.texts = new_texts_df

    return new_dset


def _get_studies_to_keep(dset, feature_group, min_img_thr=20, freq_thr=0.05):
    feature_names = dset.annotations.columns.values
    feature_names = [f for f in feature_names if f.startswith(feature_group)]

    # Get topics with more than 10 images and at least 5% of the dataset
    feature_to_keep_ids = []
    feature_to_keep = []
    idx_to_keep = []
    for f_i, feature in enumerate(feature_names):
        temp_feature_ids = dset.get_studies_by_label(
            labels=[feature],
            label_threshold=freq_thr,
        )
        if len(temp_feature_ids) >= min_img_thr:
            idx_to_keep.append(f_i)
            feature_to_keep.append(feature)
            feature_to_keep_ids.append(temp_feature_ids)

    return idx_to_keep, feature_to_keep, feature_to_keep_ids


def get_pmcids_from_dois(dois):
    """Query PubMed for the PMC IDs of a list of papers based on their DOIs."""
    pmids = []
    for i in range(0, len(dois), 100):
        chunk = dois[i : i + 100]
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=({' OR '.join(chunk)})&retmode=json"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"PubMed API returned status code {response.status_code} for {url}")
        data = response.json()
        pmids += data["esearchresult"]["idlist"]
    pmcids = []
    for i in range(0, len(pmids), 100):
        chunk = pmids[i : i + 100]
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={'%2C'.join(chunk)}&format=json"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"PubMed API returned status code {response.status_code} for {url}")
        data = response.json()
        for record in data["records"]:
            if "pmcid" in record:
                pmcids.append(record["pmcid"])
    return pmcids
