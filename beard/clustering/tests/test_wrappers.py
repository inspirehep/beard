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

from beard.metrics import silhouette_score
from functools import partial


# Test wrapper of Scipy hierarchical clustering
@pytest.fixture
def test_data():
    """Preparing a test data."""
    random_state = check_random_state(42)

    return make_blobs(centers=4, shuffle=False, random_state=random_state)


@pytest.fixture
def affinitize(test_data):
    """Create an affinity matrix Xa from X."""
    X, y = test_data
    d = euclidean_distances(X)
    d = (d + d.T) / 2.0
    d /= d.max()
    Xa = d
    return Xa


@pytest.fixture
def affinity_data(affinitize):
    """Preparing a test data where the input is an affinity matrix."""
    return affinitize


@pytest.fixture
def unsupervised_r(test_data):
    """Preparing Xr (input data in raw format) for unsupervised cases."""
    Xr, _ = test_data
    return Xr


@pytest.fixture
def supervised_a(test_data, affinitize):
    """Preparing Xa (input data in affinity format) for supervised cases."""
    random_state = check_random_state(42)
    X, y = test_data
    mask = random_state.randint(2, size=len(y)).astype(np.bool)
    y[mask] = -1
    Xa = affinitize
    return Xa, y


@pytest.fixture
def supervised_r(test_data):
    """Preparing Xr (input data in raw format) for supervised cases."""
    random_state = check_random_state(42)
    Xr, y = test_data
    mask = random_state.randint(2, size=len(y)).astype(np.bool)
    y[mask] = -1
    return Xr, y


def _ground_truth_Xr_score(Xr, labels_true, pred_labels):
    """Scoring function that takes exactly 3 arguments."""
    assert Xr.shape[0] != Xr.shape[1]  # Check that it's not an affinity matrix
    # For the sake of the test cases, X will be ignored. So, it will be like
    # using a scoring function that takes exactly 2 arguments.
    score = b3_f_score(labels_true, pred_labels)
    return score


def _ground_truth_Xa_score(Xa, labels_true, pred_labels):
    """Scoring function that takes exactly 3 arguments."""
    assert Xa.shape[0] == Xa.shape[1]  # Check that it is an affinity matrix
    # For the sake of the test cases, X will be ignored. So, it will be like
    # using a scoring function that takes exactly 2 arguments.
    score = b3_f_score(labels_true, pred_labels)
    return score


def test_scipy_HAC_semi_supervised_labels_scoring_data_affinity(supervised_a):
    """Test semi-supervised learning for sp.HAC when scoring_data=affinity."""
    X, y = supervised_a

    # We should find all 4 clusters
    clusterer = ScipyHierarchicalClustering(scoring=_ground_truth_Xa_score,
                                            scoring_data="affinity",
                                            affinity=euclidean_distances)
    clusterer.fit(X, y)
    labels = clusterer.labels_
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_HAC_semi_supervised_labels_scoring_data_raw(supervised_r):
    """Test semi-supervised learning for sp.HAC when scoring_data=raw."""
    X, y = supervised_r

    # We should find all 4 clusters
    clusterer = ScipyHierarchicalClustering(scoring=_ground_truth_Xr_score,
                                            scoring_data="raw")
    clusterer.fit(X, y)
    labels = clusterer.labels_
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_HAC_semi_supervised_labels_scoring_data_None(supervised_a):
    """Test semi-supervised learning for Scipy hierarchical clustering."""
    X, y = supervised_a

    # We should find all 4 clusters
    clusterer = ScipyHierarchicalClustering(scoring=b3_f_score)
    clusterer.fit(X, y)
    labels = clusterer.labels_
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_HAC_unsupervised_scoring_data_affinity(affinitize):
    """Unsupervised clustering when scoring_data is affinity."""
    X = affinitize
    assert X.shape[0] == X.shape[1]  # Check that it is an affinity matrix
    scoring = partial(silhouette_score, metric="precomputed")
    clusterer = ScipyHierarchicalClustering(affinity=euclidean_distances,
                                            scoring=scoring,
                                            scoring_data="affinity")
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_HAC_unsupervised_scoring_data_raw(unsupervised_r):
    """Unsupervised clustering when scoring_data is raw."""
    X = unsupervised_r
    assert X.shape[0] != X.shape[1]  # Check that it's not an affinity matrix
    scoring = partial(silhouette_score, metric="euclidean")
    clusterer = ScipyHierarchicalClustering(affinity=euclidean_distances,
                                            scoring=scoring,
                                            scoring_data="raw")
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_HAC_unsupervised_scoring_data_None(affinitize):
    """Unsupervised clustering when scoring_data is None."""
    X = affinitize

    def _pred_labels_score(pred_labels):
        """Scoring function that takes exactly 1 argument: labels."""
        return -np.inf

    clusterer = ScipyHierarchicalClustering(affinity=euclidean_distances,
                                            scoring=_pred_labels_score)
    labels = clusterer.fit_predict(X)
    assert_array_equal([100], np.bincount(labels))


def test_scipy_HAC_default_euclidean(affinity_data):
    """Default parameters, using euclidean distance."""
    X = affinity_data
    clusterer = ScipyHierarchicalClustering(n_clusters=4)
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_HAC_custom_affinity(test_data):
    """Using custom affinity function."""
    X, y = test_data
    clusterer = ScipyHierarchicalClustering(affinity=euclidean_distances,
                                            scoring_data="affinity",
                                            n_clusters=4)
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_HAC_precomputed_distance(affinity_data):
    """Using precomputed distances."""
    X = affinity_data
    clusterer = ScipyHierarchicalClustering(affinity="precomputed",
                                            scoring_data="affinity",
                                            n_clusters=4)
    labels = clusterer.fit_predict(X)
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_HAC_number_clusters(affinity_data):
    """Changing number of clusters."""
    clusterer = ScipyHierarchicalClustering(affinity="precomputed",
                                            scoring_data="affinity",
                                            n_clusters=4)
    X = affinity_data

    labels = clusterer.fit_predict(X)
    clusterer.set_params(n_clusters=10)
    labels = clusterer.labels_
    assert_equal(len(np.unique(labels)), 10)


def test_scipy_HAC_threshold(affinity_data):
    """Changing threshold."""
    clusterer = ScipyHierarchicalClustering(affinity="precomputed",
                                            scoring_data="affinity",
                                            n_clusters=4)
    X = affinity_data

    labels = clusterer.fit_predict(X)
    clusterer.set_params(threshold=clusterer.linkage_[-4,  2])
    labels = clusterer.labels_
    assert_array_equal([25, 25, 25, 25], np.bincount(labels))


def test_scipy_HAC_validation(affinity_data):
    """Test the validation of hyper-parameters and input data."""
    X = affinity_data

    with pytest.raises(ValueError):
        clusterer = ScipyHierarchicalClustering(n_clusters=len(X) + 1)
        labels = clusterer.fit_predict(X)

    with pytest.raises(ValueError):
        clusterer = ScipyHierarchicalClustering(n_clusters=-1)
        labels = clusterer.fit_predict(X)

    with pytest.raises(ValueError):
        clusterer = ScipyHierarchicalClustering(scoring_data="affinity")
        labels = clusterer.fit_predict(X)
