# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Test of blocking for clustering.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
import pytest

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state

from ..blocking import BlockClustering
from ...metrics import paired_f_score


def test_fit():
    """Test fit."""
    random_state = check_random_state(42)
    X, y = make_blobs(centers=4, shuffle=False, random_state=random_state)

    # Single block
    clusterer = BlockClustering(
        blocking="single",
        base_estimator=AgglomerativeClustering(n_clusters=4,
                                               linkage="complete"))
    clusterer.fit(X)

    assert_equal(len(clusterer.clusterers_), 1)
    assert_equal(len(np.unique(clusterer.labels_)), 4)

    # Precomputed blocks
    clusterer = BlockClustering(
        blocking="precomputed",
        base_estimator=AgglomerativeClustering(n_clusters=2,
                                               linkage="complete"))
    clusterer.fit(X, blocks=(y <= 1))

    assert_equal(len(clusterer.clusterers_), 2)
    assert_equal(len(np.unique(clusterer.labels_)), 4)

    # Custom blocking function
    X_ids = np.arange(len(X)).reshape((-1, 1))

    def _blocking(X_ids):
        return y[X_ids.ravel()] <= 1  # block labels into {0,1} and {2,3}

    def _distance(X_ids):
        return euclidean_distances(X[X_ids.ravel()])

    clusterer = BlockClustering(
        blocking=_blocking,
        base_estimator=AgglomerativeClustering(n_clusters=2,
                                               linkage="complete",
                                               affinity=_distance))
    clusterer.fit(X_ids)

    assert_equal(len(clusterer.clusterers_), 2)
    assert_equal(len(np.unique(clusterer.labels_)), 4)


def test_partial_fit():
    """Test partial_fit."""
    random_state = check_random_state(42)
    X, y = make_blobs(centers=4, shuffle=False, random_state=random_state)
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


def test_predict():
    """Test predict."""
    random_state = check_random_state(42)
    X, y = make_blobs(centers=4, shuffle=False, random_state=random_state)

    clusterer = BlockClustering(blocking="precomputed",
                                base_estimator=MiniBatchKMeans(n_clusters=2))
    clusterer.fit(X, blocks=(y <= 1))
    pred = clusterer.predict(X, blocks=(y <= 1))
    assert_equal(len(np.unique(pred)), 4)

    pred = clusterer.predict(X, blocks=10 * np.ones(len(X)))
    assert_array_equal(-np.ones(len(X)), pred)


def test_validation():
    """Test the validation of hyper-parameters and input data."""
    random_state = check_random_state(42)
    X, y = make_blobs(centers=4, shuffle=False, random_state=random_state)

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
