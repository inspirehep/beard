# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Tests of blocking for clustering.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

from pytest import mark
import pytest

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state

from beard.clustering import BlockClustering
from beard.clustering import ScipyHierarchicalClustering
from beard.metrics import paired_f_score

random_state = check_random_state(42)
X, y = make_blobs(centers=4, shuffle=False, random_state=random_state)


def _distance(X_ids):
    return euclidean_distances(X[X_ids.ravel()])


@mark.parametrize('n_jobs', (1, 2))
def test_fit(n_jobs):
    """Test fit."""
    # Single block
    clusterer = BlockClustering(
        blocking="single",
        base_estimator=AgglomerativeClustering(n_clusters=4,
                                               linkage="complete"),
        n_jobs=n_jobs)
    clusterer.fit(X)

    assert_equal(len(clusterer.clusterers_), 1)
    assert_array_equal([25, 25, 25, 25], np.bincount(clusterer.labels_))

    # Precomputed blocks
    clusterer = BlockClustering(
        blocking="precomputed",
        base_estimator=AgglomerativeClustering(n_clusters=2,
                                               linkage="complete"),
        n_jobs=n_jobs)
    clusterer.fit(X, blocks=(y <= 1))

    assert_equal(len(clusterer.clusterers_), 2)
    assert_array_equal([25, 25, 25, 25], np.bincount(clusterer.labels_))

    # Precomputed affinity
    clusterer = BlockClustering(
        affinity="precomputed",
        blocking="precomputed",
        base_estimator=ScipyHierarchicalClustering(affinity="precomputed",
                                                   n_clusters=2,
                                                   method="complete"),
        n_jobs=n_jobs)
    X_affinity = euclidean_distances(X)
    clusterer.fit(X_affinity, blocks=(y <= 1))

    assert_equal(len(clusterer.clusterers_), 2)
    assert_array_equal([25, 25, 25, 25], np.bincount(clusterer.labels_))

    # Custom blocking function
    X_ids = np.arange(len(X)).reshape((-1, 1))

    def _blocking(X_ids):
        return y[X_ids.ravel()] <= 1  # block labels into {0,1} and {2,3}

    clusterer = BlockClustering(
        blocking=_blocking,
        base_estimator=AgglomerativeClustering(n_clusters=2,
                                               linkage="complete",
                                               affinity=_distance))
    clusterer.fit(X_ids)

    assert_equal(len(clusterer.clusterers_), 2)
    assert_array_equal([25, 25, 25, 25], np.bincount(clusterer.labels_))


def test_partial_fit():
    """Test partial_fit."""
    blocks = (y <= 1)

    clusterer1 = BlockClustering(blocking="precomputed",
                                 base_estimator=MiniBatchKMeans(n_clusters=2))
    clusterer1.partial_fit(X[y <= 1], blocks=blocks[y <= 1])
    assert_equal(len(clusterer1.clusterers_), 1)
    clusterer1.partial_fit(X[y > 1], blocks=blocks[y > 1])
    assert_equal(len(clusterer1.clusterers_), 2)

    clusterer2 = BlockClustering(blocking="precomputed",
                                 base_estimator=MiniBatchKMeans(n_clusters=2))
    clusterer2.fit(X, blocks=blocks)

    c1 = clusterer1.predict(X, blocks=blocks)
    c2 = clusterer2.labels_

    assert_equal(paired_f_score(c1, c2), 1.0)


def test_onthefly_labels():
    """Test assigning labels on the fly."""
    clusterer = BlockClustering(
        base_estimator=ScipyHierarchicalClustering(n_clusters=1,
                                                   method="complete"))
    clusterer.fit(X)
    assert_array_equal([100], np.bincount(clusterer.labels_))
    clusterer.clusterers_[0].set_params(n_clusters=4)
    assert_array_equal([25, 25, 25, 25], np.bincount(clusterer.labels_))


def test_predict():
    """Test predict."""
    clusterer = BlockClustering(blocking="precomputed",
                                base_estimator=MiniBatchKMeans(n_clusters=2))
    clusterer.fit(X, blocks=(y <= 1))
    pred = clusterer.predict(X, blocks=(y <= 1))
    assert_array_equal([25, 25, 25, 25], np.bincount(clusterer.labels_))

    pred = clusterer.predict(X, blocks=10 * np.ones(len(X)))
    assert_array_equal(-np.ones(len(X)), pred)


@mark.parametrize('n_jobs', (1, 2))
def test_single_signature(n_jobs):
    """Test clustering of a  single signature."""
    import numbers
    clusterer = BlockClustering(base_estimator=MiniBatchKMeans(n_clusters=2))
    clusterer.fit(np.array([X[0]]))
    assert isinstance(clusterer.predict(X[0])[0], numbers.Integral)


def test_validation():
    """Test the validation of hyper-parameters and input data."""
    with pytest.raises(ValueError):
        clusterer = BlockClustering(
            blocking="foobar",
            base_estimator=MiniBatchKMeans(n_clusters=2))
        clusterer.fit(X)

    with pytest.raises(ValueError):
        clusterer = BlockClustering(
            blocking="precomputed",
            base_estimator=MiniBatchKMeans(n_clusters=2))
        clusterer.fit(X)

    with pytest.raises(ValueError):
        clusterer = BlockClustering(
            blocking="precomputed",
            base_estimator=MiniBatchKMeans(n_clusters=2))
        clusterer.fit(X, blocks=(y <= 1))
        clusterer.predict(X)
