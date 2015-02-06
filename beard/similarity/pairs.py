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
import numpy as np
import scipy.sparse as sp

from sklearn.base import TransformerMixin


class PairTransformer(TransformerMixin):

    """Apply a transformer on all elements in paired data."""

    def __init__(self, element_transformer):
        """Initialize.

        Parameters
        ----------
        :param element_transformer: transformer
            The transformer to apply on each element.
        """
        self.element_transformer = element_transformer

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
        n_samples, n_features_all = X.shape
        self.n_features_ = n_features_all / 2

        if sp.issparse(X):
            Xt = sp.vstack((X[:, :self.n_features_],
                            X[:, self.n_features_:])).tocsr()
        else:
            Xt = np.vstack((X[:, :self.n_features_], X[:, self.n_features_:]))

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
        n_samples = X.shape[0]

        if sp.issparse(X):
            Xt = sp.vstack((X[:, :self.n_features_],
                            X[:, self.n_features_:])).tocsr()
        else:
            Xt = np.vstack((X[:, :self.n_features_], X[:, self.n_features_:]))

        Xt = self.element_transformer.transform(Xt)

        if sp.issparse(Xt):
            Xt = sp.hstack((Xt[:n_samples], Xt[n_samples:])).tocsr()
        else:
            Xt = np.hstack((Xt[:n_samples], Xt[n_samples:]))

        return Xt


class CosineSimilarity(TransformerMixin):

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
        n_features = n_features_all / 2
        sparse = sp.issparse(X)

        if sparse:
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


class AbsoluteDifference(TransformerMixin):

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
        n_features = n_features_all / 2

        if sp.issparse(X):
            X = X.todense()

        X1 = X[:, :n_features]
        X2 = X[:, n_features:]

        return np.abs(X1 - X2)
