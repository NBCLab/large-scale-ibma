"""Modiffied LDA model from NiMARE"""

import numpy as np
import pandas as pd
from nimare.base import NiMAREBase
from nimare.utils import _check_ncores
from sklearn.decomposition import LatentDirichletAllocation


def _annotate_dset(dataset, model, counts_df, doc_topic_weights):
    """Annotate a dataset with LDA model transformed weights.

    Parameters
    ----------
    dataset : :obj:`nimare.dataset.Dataset`
        The dataset to be annotated.
    model : :obj:`sklearn.decomposition.LatentDirichletAllocation`
        The LDA model used to transform the dataset.
    counts_df : :obj:`pandas.DataFrame`
        The counts used to train the LDA model.
    doc_topic_weights : :obj:`numpy.ndarray`
        The transformed weights for the dataset.

    Returns
    -------
    new_dset : :obj:`nimare.dataset.Dataset`
        The annotated dataset.
    """
    vocabulary = counts_df.columns.to_numpy()
    study_ids = counts_df.index.tolist()

    topic_word_weights = model.components_
    n_topics = model.n_components

    sorted_weights_idxs = np.argsort(-topic_word_weights, axis=1)
    top_tokens = [
        "_".join(vocabulary[sorted_weights_idxs[topic_i, :]][:3]) for topic_i in range(n_topics)
    ]
    topic_names = [f"LDA{n_topics}__{i + 1}_{top_tokens[i]}" for i in range(n_topics)]

    doc_topic_weights_df = pd.DataFrame(
        index=study_ids,
        columns=topic_names,
        data=doc_topic_weights,
    )

    annotations = dataset.annotations.copy()
    annotations = pd.merge(annotations, doc_topic_weights_df, left_on="id", right_index=True)
    new_dset = dataset.copy()
    new_dset.annotations = annotations

    return new_dset


class LDAModel(NiMAREBase):
    """Generate a latent Dirichlet allocation (LDA) topic model.

    This class is a light wrapper around scikit-learn tools for tokenization and LDA.

    Parameters
    ----------
    n_topics : :obj:`int`
        Number of topics for topic model. This corresponds to the model's ``n_components``
        parameter. Must be an integer >= 1.
    max_iter : :obj:`int`, optional
        Maximum number of iterations to use during model fitting. Default = 1000.
    alpha : :obj:`float` or None, optional
        The ``alpha`` value for the model. This corresponds to the model's ``doc_topic_prior``
        parameter. Default is None, which evaluates to ``1 / n_topics``,
        as was used in :footcite:t:`poldrack2012discovering`.
    beta : :obj:`float` or None, optional
        The ``beta`` value for the model. This corresponds to the model's ``topic_word_prior``
        parameter. If None, it evaluates to ``1 / n_topics``.
        Default is 0.001, which was used in :footcite:t:`poldrack2012discovering`.
    text_column : :obj:`str`, optional
        The source of text to use for the model. This should correspond to an existing column
        in the :py:attr:`~nimare.dataset.Dataset.texts` attribute. Default is "abstract".
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        Default is 1.

    Attributes
    ----------
    model : :obj:`~sklearn.decomposition.LatentDirichletAllocation`

    Notes
    -----
    Adapted from: https://github.com/neurostuff/NiMARE/blob/main/nimare/annotate/lda.py.

    Latent Dirichlet allocation was first developed in :footcite:t:`blei2003latent`,
    and was first applied to neuroimaging articles in :footcite:t:`poldrack2012discovering`.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`~sklearn.feature_extraction.text.CountVectorizer`: Used to build a vocabulary of terms
        and their associated counts from texts in the ``self.text_column`` of the Dataset's
        ``texts`` attribute.
    :class:`~sklearn.decomposition.LatentDirichletAllocation`: Used to train the LDA model.
    """

    def __init__(
        self, n_topics, max_iter=1000, alpha=None, beta=0.001, text_column="abstract", n_cores=1
    ):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.text_column = text_column
        self.n_cores = _check_ncores(n_cores)

        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method="batch",
            doc_topic_prior=alpha,
            topic_word_prior=beta,
            n_jobs=n_cores,
        )

    def fit(self, dset, counts_df):
        """Fit the LDA topic model to text from a Dataset.

        Parameters
        ----------
        dset : :obj:`~nimare.dataset.Dataset`
            A Dataset with, at minimum, text available in the ``self.text_column`` column of its
            :py:attr:`~nimare.dataset.Dataset.texts` attribute.
        count_df : :obj:`pandas.DataFrame`
            A DataFrame with feature counts for the model. The index is 'id',
            used for identifying studies. Other columns are features (e.g.,
            unigrams and bigrams from Neurosynth), where each value is the number
            of times the feature is found in a given article.

        Returns
        -------
        dset : :obj:`~nimare.dataset.Dataset`
            A new Dataset with an updated :py:attr:`~nimare.dataset.Dataset.annotations` attribute.

        Attributes
        ----------
        distributions_ : :obj:`dict`
            A dictionary containing additional distributions produced by the model, including:

                -   ``p_topic_g_word``: :obj:`numpy.ndarray` of shape (n_topics, n_tokens)
                    containing the topic-term weights for the model.
                -   ``p_topic_g_word_df``: :obj:`pandas.DataFrame` of shape (n_topics, n_tokens)
                    containing the topic-term weights for the model.
        """
        count_values = counts_df.values

        doc_topic_weights = self.model.fit_transform(count_values)

        return _annotate_dset(dset, self.model, counts_df, doc_topic_weights)


def annotate_lda(dataset, counts_df, n_topics=100, max_iter=1000, n_cores=1):
    """Annotate Dataset with the resutls of an LDA model.

    Parameters
    ----------
    dset : :obj:`~nimare.dataset.Dataset`
        A Dataset with, at minimum, text available in the ``self.text_column`` column of its
        :py:attr:`~nimare.dataset.Dataset.texts` attribute.
    n_topics : :obj:`int`
        Number of topics for topic model. This corresponds to the model's ``n_components``
        parameter. Must be an integer >= 1.
    dset_name: str
        Dataset name. Possible options: "neurosynth" or "neuroquery"
    data_dir: str
        Path to data directory.
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        Default is 1.

    Returns
    -------
    dset : :obj:`~nimare.dataset.Dataset`
        A new Dataset with an updated :py:attr:`~nimare.dataset.Dataset.annotations` attribute.
    """
    model = LDAModel(n_topics=n_topics, max_iter=max_iter, n_cores=n_cores)
    dataset = model.fit(dataset, counts_df)

    return dataset, model
