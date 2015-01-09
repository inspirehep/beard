# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Blocking for clustering estimators.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin
from sklearn.base import clone
from sklearn.utils import column_or_1d


def _single(X):
    return np.ones(len(X), dtype=np.int)


class BlockClustering(BaseEstimator, ClusterMixin):

    """Implements blocking for clustering estimators.

    Meta-estimator for grouping samples into blocks, within each of which
    a clustering base estimator is fit. This allows to reduce the cost of
    pairwise distance computation from O(N^2) to O(sum_b N_b^2), where
    N_b <= N is the number of samples in block b.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.
    """

    def __init__(self, blocking="single", base_estimator=None):
        """Initialize.

        Parameters
        ----------
        :param blocking: string or callable, default "single"
            The blocking strategy, for mapping samples X to blocks.
            - "single": group all samples X[i] into the same block;
            - "precomputed": use `blocks[i]` argument (in `fit`, `partial_fit`
              or `predict`) a key for mapping sample X[i] to a block;
            - callable: use blocking(X)[i] as a key for mapping sample X[i] to
              a block.

        :param base_estimator: estimator
            Clustering estimator to fit within each block.
        """
        self.blocking = blocking
        self.base_estimator = base_estimator

    def _validate(self, X, blocks):
        """Validate hyper-parameters and input data. """
        if self.blocking == "single":
            blocks = _single(X)
        elif self.blocking == "precomputed":
            if blocks is not None and len(blocks) == len(X):
                blocks = column_or_1d(blocks)
            else:
                raise ValueError("Invalid value for blocks. When "
                                 "blocking='precomputed', blocks needs to be "
                                 "an array of size len(X).")
        elif callable(self.blocking):
            blocks = self.blocking(X)
        else:
            raise ValueError("Invalid value for blocking. Allowed values are "
                             "'single', 'precomputed' or callable.")

        return X, blocks

    def _fit(self, X, blocks):
        """Fit base clustering estimators on X."""
        self.labels_ = -np.ones(len(X), dtype=np.int)
        offset = 0

        for b in np.unique(blocks):
            # Fit on the block
            mask = (blocks == b)

            if self.fit_:
                cluster = clone(self.base_estimator)
                cluster.fit(X[mask])

            elif self.partial_fit_:
                if b in self.clusterers_:
                    cluster = self.clusterers_[b]
                else:
                    cluster = clone(self.base_estimator)

                if hasattr(cluster, "partial_fit"):
                    cluster.partial_fit(X[mask])
                else:
                    cluster.fit(X[mask])

            self.clusterers_[b] = cluster

            pred = np.array(cluster.labels_)
            mask_unknown = (pred == -1)
            pred[~mask_unknown] += offset
            self.labels_[mask] = pred
            offset += np.max(cluster.labels_) + 1

        return self

    def fit(self, X, y=None, blocks=None):
        """Fit individual base clustering estimators for each block.

        Parameters
        ----------
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        :param blocks: array-like, shape (n_samples, )
            Block labels, if `blocking == 'precomputed'`.

        Returns
        -------
        :param self: self
        """
        # Validate parameters
        X, blocks = self._validate(X, blocks)

        # Reset attributes
        self.clusterers_ = {}
        self.fit_, self.partial_fit_ = True, False

        return self._fit(X, blocks)

    def partial_fit(self, X, y=None, blocks=None):
        """Resume fitting of base clustering estimators, for each block.

        This calls `partial_fit` whenever supported by the base estimator.
        Otherwise, this calls `fit`, on given blocks only.

        Parameters
        ----------
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        :param blocks: array-like, shape (n_samples, )
            Block labels, if `blocking == 'precomputed'`.

        Returns
        -------
        :param self: self
        """
        # Validate parameters
        X, blocks = self._validate(X, blocks)

        # Set attributes if first call
        if not hasattr(self, "clusterers_"):
            self.clusterers_ = {}

        self.fit_, self.partial_fit_ = False, True

        return self._fit(X, blocks)

    def predict(self, X, blocks=None):
        """Predict data.

        Parameters
        ----------
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        :param blocks: array-like, shape (n_samples, )
            Block labels, if `blocking == 'precomputed'`.

        Returns
        -------
        :param labels: array-like, shape (n_samples)
            The labels.
        """
        # Validate parameters
        X, blocks = self._validate(X, blocks)

        # Predict
        labels = -np.ones(len(X), dtype=np.int)
        offset = 0

        for b in np.unique(blocks):
            # Predict on the block, if known
            if b in self.clusterers_:
                mask = (blocks == b)
                cluster = self.clusterers_[b]

                pred = np.array(cluster.predict(X[mask]))
                mask_unknown = (pred == -1)
                pred[~mask_unknown] += offset
                labels[mask] = pred
                offset += np.max(cluster.labels_) + 1

        return labels
