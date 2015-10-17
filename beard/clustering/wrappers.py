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
.. codeauthor:: Hussein Al-Natsheh <h.natsheh@ciapple.com>

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
                 depth=2, R=None, monocrit=None, unsupervised_scoring=None,
                 supervised_scoring=None, scoring_data=None,
                 best_threshold_precedence=True):
        """Initialize.

        Parameters
        ----------
        :param method: string
            The linkage algorithm to use.
            See scipy.cluster.hierarchy.linkage for further details.

        :param affinity: string or callable
            The distance metric to use.
            - "precomputed": assume that X is a distance matrix;
            - callable: a function returning a distance matrix.
            - Otherwise, any value supported by
              scipy.cluster.hierarchy.linkage.

        :param n_clusters: int
            The number of flat clusters to form.

        :param threshold: float or None
            The thresold to apply when forming flat clusters, if
            n_clusters=None.
            See scipy.cluster.hierarchy.fcluster for further details.

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

         :param scoring_data: string or None
            The type of input data to pass to the scoring function:
            - "raw": for passing the original X array;
            - "affinity": for passing an affinity matrix Xa;
            - None: for not passing anything but the labels.

        :param supervised_scoring: callable or None
            The scoring function to maximize in order to estimate the best
            threshold. Labels must be provided in y for this scoring function.
            There are 3 possible cases based on the value of `scoring_data`:
                - scoring_data == "raw":
                    supervised_scoring(X_raw, label_true, label_pred);
                - scoring_data == "affinity":
                    supervised_scoring(X_affinity, label_true, label_pred);
                - scoring_data is None:
                    supervised_scoring(label_true, label_pred).

        :param unsupervised_scoring: callable or None
            The scoring function to maximize in order to estimate the best
            threshold.  Labels must not be provided in y for this scoring
            function.There are 3 possible cases based on the value of
            `scoring_data`:
                - scoring_data == "raw":
                    unsupervised_scoring(X_raw, label_pred);
                - scoring_data == "affinity":
                    unsupervised_scoring(X_affinity, label_pred);
                - scoring_data is None:
                    unsupervised_scoring(label_pred).

        """
        self.method = method
        self.affinity = affinity
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.criterion = criterion
        self.depth = depth
        self.R = R
        self.monocrit = monocrit
        self.unsupervised_scoring = unsupervised_scoring
        self.supervised_scoring = supervised_scoring
        self.scoring_data = scoring_data
        self.best_threshold_precedence = best_threshold_precedence

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
        X_raw = X
        n_samples = X.shape[0]

        # Build linkage matrix
        if self.affinity == "precomputed" or callable(self.affinity):
            if callable(self.affinity):
                X = self.affinity(X)
            X_affinity = X
            if X.ndim == 2:
                i, j = np.triu_indices(X.shape[0], k=1)
                X = X[i, j]
            self.linkage_ = hac.linkage(X, method=self.method)
        else:
            X_affinity = None
            self.linkage_ = hac.linkage(X,
                                        method=self.method,
                                        metric=self.affinity)

        if self.scoring_data == "affinity" and X_affinity is None:
            raise ValueError("The scoring function expects an affinity matrix,"
                             " which cannot be computed from the combination"
                             " of parameters you provided.")

        # Estimate threshold in case of semi-supervised or unsupervised
        # As default value we use the highest so we obtain only 1 cluster.
        best_threshold = (self.linkage_[-1, 2] if self.threshold is None
                          else self.threshold)

        n_clusters = self.n_clusters
        supervised_scoring = self.supervised_scoring
        unsupervised_scoring = self.unsupervised_scoring
        ground_truth = (y is not None) and np.any(np.array(y) != -1)
        scoring = supervised_scoring is not None or \
            unsupervised_scoring is not None

        if n_clusters is None and scoring:
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

                if ground_truth and supervised_scoring is not None:
                    train = (y != -1)

                    if self.scoring_data == "raw":
                        score = supervised_scoring(X_raw, y[train],
                                                   labels[train])

                    elif self.scoring_data == "affinity":
                        score = supervised_scoring(X_affinity, y[train],
                                                   labels[train])

                    else:
                        score = supervised_scoring(y[train],
                                                   labels[train])

                elif unsupervised_scoring is not None:
                    if self.scoring_data == "raw":
                        score = unsupervised_scoring(X_raw, labels)

                    elif self.scoring_data == "affinity":
                        score = unsupervised_scoring(X_affinity, labels)

                    else:
                        score = unsupervised_scoring(labels)

                else:
                    break

                if score >= best_score:
                    best_score = score
                    best_threshold = threshold

        self.best_threshold_ = best_threshold
        self.n_samples_ = n_samples

        return self

    @property
    def labels_(self):
        """Compute the labels assigned to the input data.

        Note that labels are computed on-the-fly from the linkage matrix,
        based on the value of self.threshold or self.n_clusters.
        """
        n_clusters = self.n_clusters

        if n_clusters is not None:
            if n_clusters < 1 or n_clusters > self.n_samples_:
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

                    if len(np.unique(labels)) <= n_clusters:
                        _, labels = np.unique(labels, return_inverse=True)
                        return labels

        else:
            threshold = self.threshold

            if self.best_threshold_precedence:
                threshold = self.best_threshold_

            labels = hac.fcluster(self.linkage_, threshold,
                                  criterion=self.criterion, depth=self.depth,
                                  R=self.R, monocrit=self.monocrit)

            _, labels = np.unique(labels, return_inverse=True)

            return labels
