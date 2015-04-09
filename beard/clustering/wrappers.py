# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Scikit-Learn compatible wrappers of clustering algorithms.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Hussein AL-NATSHEH <h.natsheh@ciapple.com>

"""
import numpy as np

import scipy.cluster.hierarchy as hac

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

    def __init__(self, method="single", affinity="euclidean",
                 threshold=None, n_clusters=None, criterion="distance",
                 depth=2, R=None, monocrit=None, scoring=None,
                 affinity_score=False):
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
            The thresold to apply when forming flat clusters. In case
            of semi-supervised clustering, this value is overridden by
            the threshold maximizing the provided scoring function on
            the labeled samples.
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

        :param scoring: callable
            The scoring function to maximize in order to estimate the best
            threshold. There are 4 possibles cases based on data availability:
            - ground_truth and affinity: scoring(X, labels_true, labels_pred)
            - ground_truth but not affinity: scoring(labels_true, labels_pred)
            - affinity but not ground_truth: scoring(X, labels_pred)
            - none: scoring(labels_pred).

        :param affinity_score: boolean
            A flag that must be True if the scoring function requires the
            affinity as an input. False otherwise.
        """
        self.method = method
        self.affinity = affinity
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.criterion = criterion
        self.depth = depth
        self.R = R
        self.monocrit = monocrit
        self.scoring = scoring
        self.affinity_score = affinity_score

    def fit(self, X, y=None):
        """Perform hierarchical clustering on input data.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features) or
                  (n_samples, n_samples)
            Input data, as an array of samples or as a distance matrix if
            affinity == 'precomputed'.

        :param y: array-like, shape (n_samples, )
            Input labels, in case of (semi-)supervised clustering.
            Labels equal to -1 stand for unknown labels.

        Returns
        -------
        :returns: self
        """
        X = np.array(X)
        size = X.shape[0]

        # Build linkage matrix
        if self.affinity == "precomputed":
            Xs = X
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
            self.linkage_ = hac.linkage(X, method=self.method)

        elif callable(self.affinity):
            X = self.affinity(X)
            Xs = X
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
            self.linkage_ = hac.linkage(X, method=self.method)
        else:
            self.linkage_ = hac.linkage(X,
                                        method=self.method,
                                        metric=self.affinity)

        # Estimate threshold in case of semi-supervised or unsupervised.
        # As default value we use the highest so we obtain only 1 cluster.
        best_threshold = self.linkage_[-1, 2]

        if y is not None:
            y_arr = np.array(y)
            all_y_neg = y_arr.sum() == len(y_arr) * -1
            ground_truth = y is not None and not all_y_neg
        else:
            ground_truth = False

        n_clusters = self.n_clusters
        scoring = self.scoring
        threshold = self.threshold
        if threshold is None and n_clusters is None and scoring is not None:
            best_score = -np.inf
            thresholds = np.concatenate(([0],
                                         self.linkage_[:, 2],
                                         [self.linkage_[-1, 2]]))

            for i in range(len(thresholds) - 1):
                t1, t2 = thresholds[i:i + 2]
                threshold = (t1 + t2) / 2.0
                labels = hac.fcluster(self.linkage_, threshold,
                                      criterion=self.criterion,
                                      depth=self.depth, R=self.R,
                                      monocrit=self.monocrit)
                if ground_truth:
                    train = (y != -1)

                    if not self.affinity_score:
                        score = scoring(y[train], labels[train])
                    else:
                        score = scoring(Xs, y[train], labels[train])

                elif self.affinity_score:
                    n_labels = len(np.unique(labels))
                    n_samples = Xs.shape[0]

                    if 1 < n_labels < n_samples:
                        score = scoring(Xs, labels)
                    else:
                        score = -np.inf
                else:
                    score = scoring(labels)

                if score >= best_score:
                    best_score = score
                    best_threshold = threshold

        self.best_threshold_ = best_threshold
        self.size_ = size

        return self

    @property
    def labels_(self):
        """Compute the labels assigned to the input data.

        Note that labels are computed on-the-fly from the linkage matrix,
        based on the value of self.threshold or self.n_clusters.
        """
        n_clusters = self.n_clusters

        if n_clusters is not None:

            if n_clusters < 1 or n_clusters > self.size_:
                raise ValueError("n_clusters must be within [1; n_samples].")
            else:
                thresholds = np.concatenate(([0],
                                            self.linkage_[:, 2],
                                            [self.linkage_[-1, 2]]))

                for i in range(len(thresholds) - 1):
                    t1, t2 = thresholds[i:i + 2]
                    threshold = (t1 + t2) / 2.0
                    labels = hac.fcluster(self.linkage_, threshold,
                                          criterion=self.criterion,
                                          depth=self.depth, R=self.R,
                                          monocrit=self.monocrit)

                    if len(np.unique(labels)) == n_clusters:
                        _, labels = np.unique(labels, return_inverse=True)
                        return labels

        threshold = self.threshold

        # Override threshold with the estimated one if it is None
        if threshold is None:
            threshold = self.best_threshold_

        if threshold is not None:
            labels = hac.fcluster(self.linkage_, threshold,
                                  criterion=self.criterion, depth=self.depth,
                                  R=self.R, monocrit=self.monocrit)

            _, labels = np.unique(labels, return_inverse=True)
            return labels

        else:
            raise ValueError("Clustering error. Check the inputs combination.")
