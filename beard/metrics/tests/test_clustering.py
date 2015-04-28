# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014, 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Test clustering evaluation metrics.

.. codeauthor:: Evangelos Tzemis <evangelos.tzemis@cern.ch>
.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal
import pytest

from beard.metrics.clustering import b3_precision_recall_fscore
from beard.metrics.clustering import b3_precision_score
from beard.metrics.clustering import b3_recall_score
from beard.metrics.clustering import b3_f_score
from beard.metrics.clustering import paired_precision_recall_fscore
from beard.metrics.clustering import paired_precision_score
from beard.metrics.clustering import paired_recall_score
from beard.metrics.clustering import paired_f_score
from beard.metrics.clustering import _cluster_samples
from beard.metrics.clustering import _general_merge_distance


def test_b3_precision_recall_fscore():
    """Test the results of b3_precision_recall_fscore."""
    # test for the border case where score maximum
    y = [1, 2, 1, 3, 2, 4, 5, 4]
    assert_equal(b3_precision_recall_fscore(y, y), (1, 1, 1))

    # test for border case when predicting singletons
    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 3, 4]
    assert_equal(b3_precision_recall_fscore(y_true, y_pred), (1, 0.5, 2 / 3))


def test_b3_precision_score():
    """Test the returned results of b3_precision_score."""
    y_true = [1, 1, 2, 2, 3, 4, 5]
    y_pred = [1, 2, 2, 2, 3, 4, 5]
    assert_almost_equal(b3_precision_score(y_true, y_pred), 17 / 21)

    y_true = [1, 1, 1, 4, 5, 5, 0, 4]
    y_pred = [1, 1, 1, 1, 5, 5, 6, 7]
    assert_equal(b3_precision_score(y_true, y_pred), 13 / 16)

    # test for the trivial maximum case
    assert_equal(b3_precision_score(y_true, y_true), 1)


def test_b3_recall_score():
    """Test the returned results of b3_recall_score."""
    y_true = [1, 1, 2, 2, 3, 4, 5]
    y_pred = [1, 2, 2, 2, 3, 4, 5]
    assert_almost_equal(b3_recall_score(y_true, y_pred), 6 / 7)

    y_true = [1, 1, 1, 4, 5, 5, 0, 4]
    y_pred = [1, 1, 1, 1, 5, 5, 6, 7]
    assert_equal(b3_recall_score(y_true, y_pred), 7 / 8)

    # test for the trivial maximum case
    assert_equal(b3_recall_score(y_true, y_true), 1)


def test_b3_f_score():
    """Test the returned results of b3_f_score."""
    y_true = [1, 1, 2, 2, 3, 4, 5]
    y_pred = [1, 2, 2, 2, 3, 4, 5]
    desired_output = 2 * (17 / 21) * (6 / 7) / (17 / 21 + 6 / 7)
    assert_almost_equal(b3_f_score(y_true, y_pred), desired_output)

    y_true = [1, 1, 1, 4, 5, 5, 0, 4]
    y_pred = [1, 1, 1, 1, 5, 5, 6, 7]
    desired_output = 2 * (13 / 16) * (7 / 8) / (13 / 16 + 7 / 8)
    assert_almost_equal(b3_f_score(y_true, y_pred),  desired_output)

    # test for the trivial maximum case
    assert_equal(b3_f_score(y_true, y_true), 1)


def test_b3_label_invariability():
    """Test that paired P/R/F values are label invariant."""
    y = [1, 2, 1, 3, 2, 4, 5, 4]
    y_prime_invariant = [3, 6, 6, 5, 6, 2, 4, 2]
    y_prime = [2, 3, 3, 4, 3, 5, 1, 5]
    assert_equal(b3_precision_recall_fscore(y, y_prime),
                 b3_precision_recall_fscore(y, y_prime_invariant))


def test_b3_raise_error():
    """Test the raise of the ValueError exception for paired P/R/F."""
    y = np.array([1, 2, 1, 3, 2, 4, 5, 4])

    # test raise when not 1d shape
    y = y.reshape(2, 4)
    with pytest.raises(ValueError):
        b3_precision_recall_fscore(y, y)

    # test raise when different size of elements
    y = y.reshape(8, 1)
    with pytest.raises(ValueError):
        b3_precision_recall_fscore(y[1:], y[2:])

    # test error raise when labels_true is empty
    with pytest.raises(ValueError):
        b3_precision_recall_fscore(y, [])

    # test error raise when labels_pred is empty
    with pytest.raises(ValueError):
        b3_precision_recall_fscore([], y)

    # test error raise when both inputs are empty
    with pytest.raises(ValueError):
        b3_precision_recall_fscore([], [])


def test_paired_precision_recall_fscore():
    """Test the results of paired_precision_recall_fscore."""
    # test for border case where score is maximum
    y = [1, 2, 1, 3, 2, 4, 5, 4]
    assert_equal(paired_precision_recall_fscore(y, y), (1, 1, 1))

    # test for border case where score is minimum
    y_true = [1, 2, 1, 3, 2, 4, 5, 4]
    y_pred = [1, 1, 2, 2, 3, 3, 4, 4]
    assert_equal(paired_precision_recall_fscore(y_true, y_pred), (0, 0, 0))


def test_paired_precision_score():
    """Test the returned results of paired_precision_score."""
    y_true = [1, 1, 2, 2, 3, 4, 5]
    y_pred = [1, 2, 2, 2, 3, 4, 5]
    assert_almost_equal(paired_precision_score(y_true, y_pred), 1 / 3)

    y_true = [1, 1, 1, 4, 5, 5, 0, 4]
    y_pred = [1, 1, 1, 1, 5, 5, 6, 7]
    assert_equal(paired_precision_score(y_true, y_pred), 4 / 7)

    # test for the trivial maximum case
    assert_equal(paired_precision_score(y_true, y_true), 1)


def test_paired_recall_score():
    """Test the returned results of paired_recall_score."""
    y_true = [1, 1, 2, 2, 3, 4, 5]
    y_pred = [1, 2, 2, 2, 3, 4, 5]
    assert_almost_equal(paired_recall_score(y_true, y_pred), 0.5)

    y_true = [1, 1, 1, 4, 5, 5, 0, 4]
    y_pred = [1, 1, 1, 1, 5, 5, 6, 7]
    assert_equal(paired_recall_score(y_true, y_pred), 4 / 5)

    # test for the trivial maximum case
    assert_equal(paired_recall_score(y_true, y_true), 1)


def test_paired_f_score():
    """Test the returned results of paired_f_score."""
    y_true = [1, 1, 2, 2, 3, 4, 5]
    y_pred = [1, 2, 2, 2, 3, 4, 5]
    desired_output = 2 * (1 / 3) * 0.5 / (1 / 3 + 0.5)
    assert_almost_equal(paired_f_score(y_true, y_pred), desired_output)

    y_true = [1, 1, 1, 4, 5, 5, 0, 4]
    y_pred = [1, 1, 1, 1, 5, 5, 6, 7]
    desired_output = 2 * (4 / 7) * (4 / 5) / (4 / 7 + 4 / 5)
    assert_almost_equal(paired_f_score(y_true, y_pred), desired_output)

    # test for the trivial maximum case
    assert_equal(paired_f_score(y_true, y_true), 1)


def test_paired_label_invariability():
    """Test that paired P/R/F values are label invariant."""
    y = [1, 2, 1, 3, 2, 4, 5, 4]
    y_prime_invariant = [3, 6, 6, 5, 6, 2, 4, 2]
    y_prime = [2, 3, 3, 4, 3, 5, 1, 5]
    assert_equal(paired_precision_recall_fscore(y, y_prime),
                 paired_precision_recall_fscore(y, y_prime_invariant))


def test_paired_raise_error():
    """Test the raise of the ValueError exception for paired P/R/F."""
    y = np.array([1, 2, 1, 3, 2, 4, 5, 4])

    # test raise when not 1d shape
    y = y.reshape(2, 4)
    with pytest.raises(ValueError):
        paired_precision_recall_fscore(y, y)

    # test raise when different size of elements
    y = y.reshape(8, 1)
    with pytest.raises(ValueError):
        paired_precision_recall_fscore(y[1:], y[2:])

    # test error raise when labels_true is empty
    with pytest.raises(ValueError):
        paired_precision_recall_fscore(y, [])

    # test error raise when labels_pred is empty
    with pytest.raises(ValueError):
        paired_precision_recall_fscore([], y)

    # test error raise when both inputs are empty
    with pytest.raises(ValueError):
        paired_precision_recall_fscore([], [])


def test_cluster_samples():
    """Test that samples are correctly seperated into appropriate groups."""
    y = [1, 2, 1, 3, 2, 4, 5, 4]
    cls_true = {1: [0, 2], 2: [1, 4], 3: [3], 4: [5, 7], 5: [6]}

    assert_equal(cls_true, _cluster_samples(y))


def test_general_merge_distance():
    """Test general merge distance function."""
    y_true = np.array([1, 2, 1, 2, 1, 2])
    y_pred = [1, 1, 1, 2, 2, 2]

    # test for trivial case
    assert_equal(_general_merge_distance(y_true, y_true), 0)

    # test that fs and fm has effect on result
    zero_res = _general_merge_distance(y_true, y_pred,
                                       fm=lambda x, y: 0,
                                       fs=lambda x, y: 0)
    assert_equal(zero_res, 0)

    # test for default functions
    assert_equal(_general_merge_distance(y_true, y_pred), 4)
