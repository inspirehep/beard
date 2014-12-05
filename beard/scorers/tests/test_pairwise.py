# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Test pairwise evaluation metrics for Beard.

.. codeauthor:: Evangelos Tzemis <evangelos.tzemis@cern.ch>

"""
from __future__ import division
from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal
from .. import pairwise

import numpy as np
import pytest


def test_precision_recall_fscore():
    """Test the results of precission_recall_fscore."""
    # test for the border case where score maximum
    y = np.array([1, 2, 1, 3, 2, 4, 5, 4])
    assert pairwise.precision_recall_fscore(y, y) == (1, 1, 1)

    # test error raise when labels_true is empty
    with pytest.raises(ValueError):
        pairwise.precision_recall_fscore(y, np.array([]))

    # test error raise when labels_pred is empty
    with pytest.raises(ValueError):
        pairwise.precision_recall_fscore(np.array([]), y)

    # test error raise when both inputs are empty
    with pytest.raises(ValueError):
        pairwise.precision_recall_fscore(np.array([]), np.array([]))


def test_cluster_invariability():
    """Test that computed values are cluster invariant."""
    y = np.array([1, 2, 1, 3, 2, 4, 5, 4])
    y_prime_invariant = np.array([3, 6, 6, 5, 6, 2, 4, 2])
    y_prime = np.array([2, 3, 3, 4, 3, 5, 1, 5])
    assert_equal(pairwise.precision_recall_fscore(y, y_prime),
                 pairwise.precision_recall_fscore(y, y_prime_invariant))


def test_raise_valueError():
    """Test the raise of the ValueError exception."""
    y = np.array([1, 2, 1, 3, 2, 4, 5, 4])

    # test raise when not 1d shape
    y = y.reshape(2, 4)
    with pytest.raises(ValueError):
        pairwise.precision_recall_fscore(y, y)

    # test raise when different size of elements
    y = y.reshape(8, 1)
    with pytest.raises(ValueError):
        pairwise.precision_recall_fscore(y[1:], y[2:])


def test_precision_score():
    """Test the returned results of precision_score."""
    y_true = np.array([1, 2, 1, 3, 2, 4, 5, 4])
    y_pred = [1, 3, 3, 3, 2, 4, 5, 5]
    assert_equal(pairwise.precision_score(y_true, y_pred), 8/12)

    # test for the trivial maximum case
    assert_equal(pairwise.precision_score(y_true, y_true), 1)


def test_recall_score():
    """Test the returned results of recall_score."""
    y_true = np.array([1, 2, 1, 3, 2, 4, 5, 4])
    y_pred = [1, 3, 3, 3, 2, 4, 5, 5]
    assert_equal(pairwise.recall_score(y_true, y_pred), 8/11)

    # test for the trivial maximum case
    assert_equal(pairwise.recall_score(y_true, y_true), 1)


def test_f_score():
    """Test the returned results of f_score."""
    y_true = np.array([1, 2, 1, 3, 2, 4, 5, 4])
    y_pred = [1, 3, 3, 3, 2, 4, 5, 5]
    desired_output = 2*(8/12)*(8/11)/((8/12)+(8/11))
    assert_almost_equal(pairwise.f_score(y_true, y_pred),
                        desired_output, decimal=10)

    # test for the trivial maximum case
    assert_equal(pairwise.f_score(y_true, y_true), 1)


def test_group_samples_by_cluster_id():
    """Test that samples are correctly seperated into appropriate groups."""
    y = np.array([1, 2, 1, 3, 2, 4, 5, 4])

    gs_true = [[0, 2], [1, 4], [3], [5, 7], [6]]
    gs_true_set = set(map(frozenset, gs_true))

    gs_pred = pairwise._group_samples_by_cluster_id(y)
    gs_pred_set = set(map(frozenset, gs_pred))

    assert gs_true_set == gs_pred_set


def test_calculate_pairs():
    """Test that pairs are correctly calculated."""
    samples = [[0, 2, 8], [1, 4], [3], [5, 7], [6]]
    pairs_pred = pairwise._calculate_pairs(samples)

    pairs = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [0, 2], [0, 8],
             [2, 8], [1, 4], [5, 7]]
    pairs_true = set(map(frozenset, pairs))

    assert pairs_true == pairs_pred
