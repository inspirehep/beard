# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Scikit-Learn compatible wrappers of clustering algorithms.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""
import numpy as np

import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import squareform

from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin


class ScipyHierarchicalClustering(BaseEstimator, ClusterMixin):

    """Wrapper for Scipy's hierarchical clustering implementation.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples,)
        Array of labels assigned to the input data.

    linkage_ : ndarray
        The linkage matrix.
    """

    def __init__(self, method="single", affinity="euclidean", threshold=None,
                 n_clusters=2, criterion="distance", depth=2, R=None,
                 monocrit=None):
        """Initialize.

        Parameters
        ----------
        :param method: string
            The linkage algorithm to use.
            See scipy.cluster.hierarchy.linkage for further details.

        :param affinity: string or callable
            The distance metric to use.
            - "precomputed": assume that X is a distance matrix;
            - callable: a function returning a distance matrix;
            - Otherwise, any value supported by
              scipy.cluster.hierarchy.linkage.

        :param threshold: float or None
            The thresold to apply when forming flat clusters.
            See scipy.cluster.hierarchy.fcluster for further details.

        :param n_clusters: int
            The number of flat clusters to form, if threshold=None.

        :param criterion: string
            The criterion to use in forming flat clusters.
            See scipy.cluster.hierarchy.fcluster for further details.

        :param depth: int
            The maximum depth to perform the inconsistency calculation.
            See scipy.cluster.hierarchy.fcluster for further details.

        :param R: array-like or None
            The inconsistency matrix to use for the 'inconsistent' criterion.
            See scipy.cluster.hierarchy.fcluster for further details.

        :param monocrit: array-like or None
            The statistics upon which non-singleton i is thresholded.
            See scipy.cluster.hierarchy.fcluster for further details.
        """
        self.method = method
        self.affinity = affinity
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.criterion = criterion
        self.depth = depth
        self.R = R
        self.monocrit = monocrit

    def fit(self, X, y=None):
        """Perform hierarchical clustering on input data.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features) or
                  (n_samples, n_samples)
            Input data, as an array of samples or as a distance matrix if
            affinity == 'precomputed'.

        Returns
        -------
        :returns: self
        """
        X = np.array(X)

        if self.affinity == "precomputed":
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
            self.linkage_ = hac.linkage(X, method=self.method)

        elif callable(self.affinity):
            X = self.affinity(X)
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
            self.linkage_ = hac.linkage(X, method=self.method)

        else:
            self.linkage_ = hac.linkage(X,
                                        method=self.method,
                                        metric=self.affinity)

        return self

    @property
    def labels_(self):
        """Compute the labels assigned to the input data.

        Note that labels are computed on-the-fly from the linkage matrix,
        based on the value of self.threshold or self.n_clusters.
        """
        if self.threshold is not None:
            labels = hac.fcluster(self.linkage_, self.threshold,
                                  criterion=self.criterion, depth=self.depth,
                                  R=self.R, monocrit=self.monocrit)

            _, labels = np.unique(labels, return_inverse=True)
            return labels

        else:
            thresholds = np.concatenate(([0], self.linkage_[:, 2]))

            for i in range(len(thresholds) - 1):
                t1, t2 = thresholds[i:i + 2]

                labels = hac.fcluster(self.linkage_, (t1 + t2) / 2.0,
                                      criterion=self.criterion,
                                      depth=self.depth, R=self.R,
                                      monocrit=self.monocrit)

                if len(np.unique(labels)) == self.n_clusters:
                    _, labels = np.unique(labels, return_inverse=True)
                    return labels

            raise ValueError("n_clusters must be a value in [2, len(X)].")
