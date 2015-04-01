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
.. codeauthor:: Hussein AL-NATSHEH <h.natsheh@ciapple.com>

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
import pytest

from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state

from ..wrappers import ScipyHierarchicalClustering
from beard.metrics import b3_f_score

from sklearn.metrics import silhouette_score
from functools import partial


# Test wrapper of Scipy hierarchical clustering

@pytest.fixture
def testData():
    """Preparing a test data."""
    random_state = check_random_state(42)

    return make_blobs(centers=4, shuffle=False, random_state=random_state)


def test_unsupervised(testData):
    """Unsupervised clustering."""
    X, y = testData
    scoring = partial(silhouette_score, metric="precomputed")
    clusterer = ScipyHierarchicalClustering(affinity=euclidean_distances,
                                            scoring=scoring,
                                            affinity_score=True)
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_hierarchical_clustering_default_euclidean(testData):
    """Default parameters, using euclidean distance."""
    X, y = testData
    clusterer = ScipyHierarchicalClustering(n_clusters=4)
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_hierarchical_clustering_custom_affinity(testData):
    """Using custom affinity function."""
    X, y = testData
    clusterer = ScipyHierarchicalClustering(affinity=euclidean_distances,
                                            n_clusters=4)
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_hierarchical_clustering_precomputed_distances(testData):
    """Using precomputed distances."""
    X, y = testData
    d = euclidean_distances(X)
    d = (d + d.T) / 2.0
    d /= d.max()
    clusterer = ScipyHierarchicalClustering(affinity="precomputed",
                                            n_clusters=4)
    labels = clusterer.fit_predict(d)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_hierarchical_clustering_number_clusters(testData):
    """Changing number of clusters."""
    clusterer = ScipyHierarchicalClustering(affinity="precomputed",
                                            n_clusters=4)
    X, y = testData
    d = euclidean_distances(X)
    d = (d + d.T) / 2.0
    d /= d.max()
    labels = clusterer.fit_predict(d)
    clusterer.set_params(n_clusters=10)
    labels = clusterer.labels_
    assert_equal(len(np.unique(labels)), 10)


def test_scipy_hierarchical_clustering_threshold(testData):
    """Changing threshold."""
    clusterer = ScipyHierarchicalClustering(affinity="precomputed",
                                            n_clusters=4)
    X, y = testData
    d = euclidean_distances(X)
    d = (d + d.T) / 2.0
    d /= d.max()
    labels = clusterer.fit_predict(d)
    clusterer.set_params(threshold=clusterer.linkage_[-4,  2])
    labels = clusterer.labels_
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_hierarchical_clustering_validation(testData):
    """Test the validation of hyper-parameters and input data."""
    X, y = testData

    with pytest.raises(ValueError):
        clusterer = ScipyHierarchicalClustering(n_clusters=len(X) + 1)
        labels = clusterer.fit_predict(X)

    with pytest.raises(ValueError):
        clusterer = ScipyHierarchicalClustering(n_clusters=-1)
        labels = clusterer.fit_predict(X)


@pytest.fixture
def semiSupervised():
    """Preparing a test data for semi_Supervised cases."""
    random_state = check_random_state(42)
    X, y = make_blobs(centers=4, shuffle=False, random_state=random_state)
    mask = random_state.randint(2, size=len(y)).astype(np.bool)
    y[mask] = -1
    return X, y


def test_scipy_hierarchical_clustering_semi_supervised_labels(semiSupervised):
    """Test semi-supervised learning for Scipy hierarchical clustering."""
    X, y = semiSupervised

    # We should find all 4 clusters
    clusterer = ScipyHierarchicalClustering(scoring=b3_f_score)
    clusterer.fit(X, y)
    labels = clusterer.labels_
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))
    assert_equal(hasattr(clusterer, "best_threshold_"), True)


def test_sp_hierarchical_clustering_semi_supervised_no_label(semiSupervised):
    """Test semi-supervised Scipy hierarchical clustering unkown lables."""
    X, y = semiSupervised

    # All labels are unknown, hence it should yield a single cluster
    clusterer = ScipyHierarchicalClustering()
    y[:] = -1
    clusterer.fit(X, y)
    labels = clusterer.labels_
    assert_array_equal([100], np.bincount(labels))
