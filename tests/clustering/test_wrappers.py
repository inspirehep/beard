# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Test of clustering wrappers.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Hussein Al-Natsheh<h.natsheh@ciapple.com>

"""
from __future__ import division

from functools import partial
import numpy as np
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
import pytest

from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state

from beard.metrics import b3_f_score
from beard.metrics import silhouette_score
from beard.clustering import ScipyHierarchicalClustering


def generate_data(supervised=False, affinity=False):
    rng = check_random_state(42)
    X, y = make_blobs(centers=4, cluster_std=0.01,
                      shuffle=False, random_state=rng)

    if affinity:
        d = euclidean_distances(X)
        d = (d + d.T) / 2.0
        d /= d.max()
        X = d

    if supervised:
        mask = rng.randint(2, size=len(y)).astype(np.bool)
        y[mask] = -1

    else:
        y[:] = -1

    return X, y


def test_shc_semi_supervised_scoring_data_raw():
    """Test semi-supervised learning for SHC when scoring_data='raw'."""
    X, y = generate_data(supervised=True, affinity=False)

    def _scoring(X_raw, labels_true, labels_pred):
        assert X_raw.shape == X.shape
        score = b3_f_score(labels_true, labels_pred)
        return score

    clusterer = ScipyHierarchicalClustering(supervised_scoring=_scoring,
                                            scoring_data="raw")
    clusterer.fit(X, y)
    labels = clusterer.labels_
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_shc_semi_supervised_scoring_data_affinity():
    """Test semi-supervised learning for SHC when scoring_data='affinity'."""
    # Passing feature matrix
    X1, y1 = generate_data(supervised=True, affinity=False)

    def _scoring1(X_affinity, labels_true, labels_pred):
        assert X_affinity.shape[0] == X_affinity.shape[1]
        assert X_affinity.shape != X1.shape
        score = b3_f_score(labels_true, labels_pred)
        return score

    clusterer = ScipyHierarchicalClustering(supervised_scoring=_scoring1,
                                            scoring_data="affinity",
                                            affinity=euclidean_distances)
    clusterer.fit(X1, y1)
    labels = clusterer.labels_
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))

    # Passing affinity matrix
    X2, y2 = generate_data(supervised=True, affinity=True)

    def _scoring2(X_affinity, labels_true, labels_pred):
        assert X_affinity.shape[0] == X_affinity.shape[1]
        assert X_affinity.shape == X2.shape
        score = b3_f_score(labels_true, labels_pred)
        return score

    clusterer = ScipyHierarchicalClustering(supervised_scoring=_scoring2,
                                            scoring_data="affinity",
                                            affinity="precomputed")
    clusterer.fit(X2, y2)
    labels = clusterer.labels_
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_shc_semi_supervised_scoring_data_none():
    """Test semi-supervised learning for SHC when scoring_data is None."""
    X, y = generate_data(supervised=True, affinity=False)

    def _scoring(labels_true, labels_pred):
        score = b3_f_score(labels_true, labels_pred)
        return score

    # We should find all 4 clusters
    clusterer = ScipyHierarchicalClustering(supervised_scoring=_scoring)
    clusterer.fit(X, y)
    labels = clusterer.labels_
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_shc_unsupervised_scoring_data_raw():
    """Test unsupervised clustering for SHC when scoring_data='raw'."""
    X, _ = generate_data(supervised=False, affinity=False)
    _scoring = partial(silhouette_score, metric="euclidean")
    clusterer = ScipyHierarchicalClustering(affinity=euclidean_distances,
                                            unsupervised_scoring=_scoring,
                                            scoring_data="raw")
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_shc_unsupervised_scoring_data_affinity():
    """Test unsupervised clustering for SHC when scoring_data='affinity'."""
    # Passing feature matrix
    X, _ = generate_data(supervised=False, affinity=False)
    _scoring = partial(silhouette_score, metric="precomputed")
    clusterer = ScipyHierarchicalClustering(affinity=euclidean_distances,
                                            unsupervised_scoring=_scoring,
                                            scoring_data="affinity")
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))

    # Passing affinity matrix
    X, _ = generate_data(supervised=False, affinity=True)
    _scoring = partial(silhouette_score, metric="precomputed")
    clusterer = ScipyHierarchicalClustering(affinity="precomputed",
                                            unsupervised_scoring=_scoring,
                                            scoring_data="affinity")
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_shc_unsupervised_scoring_data_None():
    """Test unsupervised clustering for SHC when scoring_data is None."""
    X, _ = generate_data(supervised=False, affinity=False)

    def _scoring(labels_pred):
        return -np.inf

    clusterer = ScipyHierarchicalClustering(affinity=euclidean_distances,
                                            unsupervised_scoring=_scoring)
    labels = clusterer.fit_predict(X)
    assert_array_equal([100], np.bincount(labels))


def test_shc_default_euclidean():
    """Test default parameters of SHC, using euclidean distance."""
    X, _ = generate_data(supervised=False, affinity=False)
    clusterer = ScipyHierarchicalClustering(n_clusters=4)
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_shc_custom_affinity():
    """Test custom affinity function in SHC."""
    X, _ = generate_data(supervised=False, affinity=False)
    clusterer = ScipyHierarchicalClustering(affinity=euclidean_distances,
                                            n_clusters=4)
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_shc_precomputed_distance():
    """Test using precomputed distances in SHC."""
    X, _ = generate_data(supervised=False, affinity=True)
    clusterer = ScipyHierarchicalClustering(affinity="precomputed",
                                            n_clusters=4)
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_shc_n_clusters():
    """Test changing number of clusters in SHC."""
    X, _ = generate_data(supervised=False, affinity=True)

    clusterer = ScipyHierarchicalClustering(affinity="precomputed",
                                            n_clusters=4)

    labels = clusterer.fit_predict(X)
    assert_equal(len(np.unique(labels)), 4)
    clusterer.set_params(n_clusters=10)
    labels = clusterer.labels_
    assert_equal(len(np.unique(labels)), 10)


def test_shc_threshold():
    """Test changing threshold in SHC."""
    X, _ = generate_data(supervised=False, affinity=True)

    # n_clusters has precedence over threshold
    clusterer = ScipyHierarchicalClustering(affinity="precomputed",
                                            n_clusters=2)
    labels1 = clusterer.fit_predict(X)
    clusterer.set_params(threshold=clusterer.linkage_[-4, 2])
    labels2 = clusterer.labels_
    assert_array_equal(labels1, labels2)
    assert_equal(len(np.unique(labels1)), 2)

    # change threshold
    clusterer.set_params(best_threshold_precedence=False)
    clusterer.set_params(n_clusters=None,
                         threshold=clusterer.linkage_[-5, 2])
    labels = clusterer.labels_
    assert_equal(len(np.unique(labels)), 5)
    clusterer.set_params(threshold=clusterer.linkage_[-4, 2])
    labels = clusterer.labels_
    assert_equal(len(np.unique(labels)), 4)


def test_shc_validation():
    """Test the validation of hyper-parameters and input data in SHC"""
    X, _ = generate_data(supervised=False, affinity=False)

    with pytest.raises(ValueError):
        clusterer = ScipyHierarchicalClustering(n_clusters=len(X) + 1)
        labels = clusterer.fit_predict(X)

    with pytest.raises(ValueError):
        clusterer = ScipyHierarchicalClustering(n_clusters=-1)
        labels = clusterer.fit_predict(X)

    with pytest.raises(ValueError):
        clusterer = ScipyHierarchicalClustering(scoring_data="affinity")
        labels = clusterer.fit_predict(X)
