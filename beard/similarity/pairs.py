# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Transformers for paired data.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Evangelos Tzemis <evangelos.tzemis@cern.ch>

"""
import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

import operator


class PairTransformer(BaseEstimator, TransformerMixin):

    """Apply a transformer on all elements in paired data."""

    def __init__(self, element_transformer, groupby=None):
        """Initialize.

        Parameters
        ----------
        :param element_transformer: transformer
            The transformer to apply on each element.

        :param groupby: callable
            If not None, use ``groupby`` as a hash to apply
            ``element_transformer`` on unique elements only.
        """
        self.element_transformer = element_transformer
        self.groupby = groupby

    def _flatten(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1] // 2
        Xt = X

        # Shortcut, when all elements are distinct
        if self.groupby is None:
            if sp.issparse(Xt):
                Xt = sp.vstack((Xt[:, :n_features],
                                Xt[:, n_features:]))
            else:
                Xt = np.vstack((Xt[:, :n_features],
                                Xt[:, n_features:]))

            return Xt, np.arange(n_samples * 2, dtype=np.int)

        # Group by keys
        groupby = self.groupby
        indices = []        # element index -> first position in X
        key_indices = {}    # key -> first position in X

        for i, element in enumerate(Xt[:, :n_features]):
            key = groupby(element)
            if key not in key_indices:
                key_indices[key] = (i, 0)
            indices.append(key_indices[key])

        for i, element in enumerate(Xt[:, n_features:]):
            key = groupby(element)
            if key not in key_indices:
                key_indices[key] = (i, n_features)
            indices.append(key_indices[key])

        # Select unique elements, from left and right
        left_indices = {}
        right_indices = {}
        key_indices = sorted(key_indices.values())
        j = 0

        for i, start in key_indices:
            if start == 0:
                left_indices[i] = j
                j += 1
        for i, start in key_indices:
            if start == n_features:
                right_indices[i] = j
                j += 1

        if sp.issparse(Xt):
            Xt = sp.vstack((Xt[sorted(left_indices.keys()), :n_features],
                            Xt[sorted(right_indices.keys()), n_features:]))
        else:
            Xt = np.vstack((Xt[sorted(left_indices.keys()), :n_features],
                            Xt[sorted(right_indices.keys()), n_features:]))

        # Map original indices to transformed values
        flat_indices = []

        for i, start in indices:
            if start == 0:
                flat_indices.append(left_indices[i])
            else:
                flat_indices.append(right_indices[i])

        return Xt, flat_indices

    def _repack(self, Xt, indices):
        n_samples = len(indices) // 2

        if sp.issparse(Xt):
            Xt = sp.hstack((Xt[indices[:n_samples]],
                            Xt[indices[n_samples:]]))
        else:
            Xt = np.hstack((Xt[indices[:n_samples]],
                            Xt[indices[n_samples:]]))

        return Xt

    def fit(self, X, y=None):
        """Fit the given transformer on all individual elements in ``X``.

        Rows i in ``X`` are assumed to represent pairs, where
        ``X[i, :n_features]`` and ``X[i, n_features:]`` correspond to their two
        individual elements. Calling ``fit`` trains the given transformer on
        the dataset formed by all these individual elements.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2 * n_features)
            Input paired data.

        Returns
        -------
        :returns: self
        """
        Xt, _ = self._flatten(X)
        self.element_transformer.fit(Xt)
        return self

    def transform(self, X):
        """Transform all individual elements in ``X``.

        Rows i in the returned array ``Xt`` represent transformed pairs, where
        ``Xt[i, :n_features_t]`` and ``Xt[i, n_features_t:]`` correspond
        to their two individual transformed elements.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2 * n_features)
            Input paired data.

        Returns
        -------
        :returns Xt: array-like, shape (n_samples, 2 * n_features_t
            The transformed data.
        """
        Xt, indices = self._flatten(X)
        Xt = self.element_transformer.transform(Xt)
        Xt = self._repack(Xt, indices)
        return Xt


class CosineSimilarity(BaseEstimator, TransformerMixin):

    """Cosine similarity on paired data."""

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Compute the cosine similarity for all pairs of elements in ``X``.

        Rows i in ``X`` are assumed to represent pairs, where
        ``X[i, :n_features]`` and ``X[i, n_features:]`` correspond to their two
        individual elements. Calling ``transform`` computes the cosine
        similarity between these elements, i.e. that ``Xt[i]`` is the cosine of
        ``X[i, :n_features]`` and ``X[i, n_features:]``.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2 * n_features)
            Input paired data.

        Returns
        -------
        :returns Xt: array-like, shape (n_samples, 2 * n_features_prime)
            The transformed data.
        """
        n_samples, n_features_all = X.shape
        n_features = n_features_all // 2
        sparse = sp.issparse(X)

        if sparse and not sp.isspmatrix_csr(X):
            X = X.tocsr()

        X1 = X[:, :n_features]
        X2 = X[:, n_features:]

        if sparse:
            numerator = np.asarray(X1.multiply(X2).sum(axis=1)).ravel()
            norm1 = np.asarray(X1.multiply(X1).sum(axis=1)).ravel()
            norm2 = np.asarray(X2.multiply(X2).sum(axis=1)).ravel()

        else:
            numerator = (X1 * X2).sum(axis=1)
            norm1 = (X1 * X1).sum(axis=1)
            norm2 = (X2 * X2).sum(axis=1)

        denominator = (norm1 ** 0.5) * (norm2 ** 0.5)

        with np.errstate(divide="ignore"):
            Xt = numerator / denominator
            Xt[denominator == 0.0] = 0.0

        return Xt.reshape((n_samples, 1))


class AbsoluteDifference(BaseEstimator, TransformerMixin):

    """Absolute difference of paired data."""

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Compute the absolute difference for all pairs of elements in ``X``.

        Rows i in ``X`` are assumed to represent pairs, where
        ``X[i, :n_features]`` and ``X[i, n_features:]`` correspond to their
        two individual elements. Calling ``transform`` computes the absolute
        difference between these elements, i.e. that ``Xt[i]`` is the
        absolute difference of ``X[i, :n_features]`` and ``X[i, n_features:]``.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2 * n_features)
            Input paired data.

        Returns
        -------
        :returns Xt: array-like, shape (n_samples, 2 * n_features_prime)
            The transformed data.
        """
        n_samples, n_features_all = X.shape
        n_features = n_features_all // 2

        if sp.issparse(X):
            X = X.todense()

        X1 = X[:, :n_features]
        X2 = X[:, n_features:]

        return np.abs(X1 - X2)


class FuzzySimilarity(TransformerMixin):

    """Implements Fuzzy Similarity (Soft TF-IDF) metric.

    For each word in document X1 we search for partial matched tockens in
    document X2. Tokens are considered a partial match if they get a good
    score using an similarity measure. Good score is considered one being
    higher than the value of the threshold variable. After obtaining
    those partial matches along with their corresponding similarity scores,
    we compute the Soft TF_IDF measure by computing the dot product of X1,
    their partial matches and the similarity scores. Finally we normalize the
    result and we return a Soft TF-IDF score for each pair of input.
    """

    def fit(self, X, y=None):
        """Return self."""
        return self

    def transform(self, X, vocabulary, similarity, threshold):
        """Compute the fuzzy similarity for all pairs of elements in `X`.

        Rows i in `X` are assumed to represent pairs, where
        `X[i, :n_features]` and `X[i, n_features:]` correspond to their two
        individual elements. Calling `transform` computes the fuzzy similarity
        between these elements.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2 * n_features)
            Input paired data.

        :param vocabulary: dict
            From word to feature_id or opposite.

        :param similarity: function
            Similarity function which returns a number between 0 and 1.

        :param threshold: float
            Threshold parameter defines the minimum level of match - similarity
            we are trying to obtain.

        Returns
        -------
        :returns X_: array-like, shape (n_samples, 1)
            The calculated fuzzy similarity for each pair of data.
        """
        n_samples, n_features_all = X.shape
        n_features = n_features_all // 2
        sparse = sp.issparse(X)

        if sparse and not sp.isspmatrix_csr(X):
            X = X.tocsr()

        X1 = X[:, :n_features]
        X2 = X[:, n_features:]

        X1_closest = sp.csr_matrix((n_samples, n_features))
        similarities = sp.csr_matrix((n_samples, n_features))

        for row, u in enumerate(X1):
            if sparse:
                u = u.toarray()[0]
            for pos, w in enumerate(u):
                w_word = vocabulary[pos]
                if sparse:
                    words = X2.getrow(row).toarray()[0]
                else:
                    words = X2[row, :]
                # Find the closest neighbour
                pos_tupl = filter(lambda t: t[1] > 0.0, enumerate(words))
                w_i = map(lambda t: (t[0], vocabulary[t[0]]), pos_tupl)
                ind_dis = map(lambda t: (t[0], similarity(w_word, t[1])), w_i)
                n_index, max_sim = max(ind_dis, key=operator.itemgetter(1))

                # Fill in the corresponding arrays according to threshold
                if threshold is None or max_sim >= threshold:
                    X1_closest[row, pos] = X2[row, n_index]
                    similarities[row, pos] = max_sim
                else:
                    X1_closest[row, pos] = X2[row, n_index]
                    similarities[row, pos] = 0

        if not sparse:
            X1 = sp.csr_matrix(X1)
        temp = X1_closest.multiply(similarities)
        numerator = np.asarray(X1.multiply(temp).sum(axis=1)).ravel()
        norm1 = np.asarray(X1.multiply(X1).sum(axis=1)).ravel()
        norm2 = np.asarray(temp.multiply(temp).sum(axis=1)).ravel()

        denominator = (norm1 ** 0.5) * (norm2 ** 0.5)
        with np.errstate(divide="ignore"):
            Xt = numerator / denominator
            Xt[denominator == 0.0] = 0.0

        return Xt.reshape((n_samples, 1))
