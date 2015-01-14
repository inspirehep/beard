# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Test of clustering wrappers.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

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


def test_scipy_hierarchical_clustering():
    """Test wrapper of Scipy hierarchical clustering."""
    random_state = check_random_state(42)
    X, y = make_blobs(centers=4, shuffle=False, random_state=random_state)

    # Default parameters, using euclidean distance
    clusterer = ScipyHierarchicalClustering(n_clusters=4)
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))
    assert_equal(hasattr(clusterer, "best_threshold_"), False)

    # Using custom affinity function
    clusterer = ScipyHierarchicalClustering(affinity=euclidean_distances,
                                            n_clusters=4)
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))

    # Using precomputed distances
    d = euclidean_distances(X)
    d = (d + d.T) / 2.0
    d /= d.max()
    clusterer = ScipyHierarchicalClustering(affinity="precomputed",
                                            n_clusters=4)
    labels = clusterer.fit_predict(d)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))

    # Change number of clusters
    clusterer.set_params(n_clusters=10)
    labels = clusterer.labels_
    assert_equal(len(np.unique(labels)), 10)

    # Change threshold
    clusterer.set_params(threshold=clusterer.linkage_[-4,  2])
    labels = clusterer.labels_
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_hierarchical_clustering_semi_supervised():
    """Test semi-supervised learning for Scipy hierarchical clustering."""
    random_state = check_random_state(42)
    X, y = make_blobs(centers=4, shuffle=False, random_state=random_state)
    mask = random_state.randint(2, size=len(y)).astype(np.bool)
    y[mask] = -1

    clusterer = ScipyHierarchicalClustering()
    clusterer.fit(X, y)
    labels = clusterer.labels_
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))
    assert_equal(hasattr(clusterer, "best_threshold_"), True)


def test_scipy_hierarchical_clustering_validation():
    """Test the validation of hyper-parameters and input data."""
    random_state = check_random_state(42)
    X, y = make_blobs(centers=4, shuffle=False, random_state=random_state)

    with pytest.raises(ValueError):
        clusterer = ScipyHierarchicalClustering(n_clusters=len(X) + 1)
        labels = clusterer.fit_predict(X)

    with pytest.raises(ValueError):
        clusterer = ScipyHierarchicalClustering(n_clusters=-1)
        labels = clusterer.fit_predict(X)
