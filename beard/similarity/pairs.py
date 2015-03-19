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

"""
from __future__ import division

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import binarize


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


class JaccardSimilarity(BaseEstimator, TransformerMixin):

    """Jaccard similarity on paired data.

    The Jaccard similarity of two elements in a pair is defined as the
    ratio between the size of their intersection and the size of their
    union.
    """

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
        """Compute the Jaccard similarity for all pairs of elements in ``X``.

        Rows i in ``X`` are assumed to represent pairs, where
        ``X[i, :n_features]`` and ``X[i, n_features:]`` correspond to their two
        individual elements, each representing a set. Calling ``transform``
        computes the Jaccard similarity between these sets, i.e. such that
        ``Xt[i]`` is the Jaccard similarity of ``X[i, :n_features]`` and
        ``X[i, n_features:]``.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns: Xt array-like, shape (n_samples, 1)
            The transformed data.
        """
        n_samples, n_features_all = X.shape
        n_features = n_features_all // 2

        X = binarize(X)
        X1 = X[:, :n_features]
        X2 = X[:, n_features:]

        sparse = sp.issparse(X)

        if sparse and not sp.isspmatrix_csr(X):
            X = X.tocsr()

        if sparse:
            if X.data.sum() == 0:
                return np.zeros((n_samples, 1))

            numerator = np.asarray(X1.multiply(X2).sum(axis=1)).ravel()

            X_sum = X1 + X2
            X_sum.data[X_sum.data != 0.] = 1
            M = X_sum.sum(axis=1)
            A = M.getA()
            denominator = A.reshape(-1,)

        else:
            if len(X[X.nonzero()]) == 0.:
                return np.zeros((n_samples, 1))

            numerator = (X1 * X2).sum(axis=1)

            X_sum = X1 + X2
            X_sum[X_sum.nonzero()] = 1
            denominator = X_sum.sum(axis=1)

        with np.errstate(divide="ignore"):
            Xt = numerator / denominator
            Xt[np.where(denominator == 0)[0]] = 0.

        return np.array(Xt).reshape(-1, 1)
