# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Tests of transformers for paired data.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

import numpy as np
from numpy.testing import assert_array_almost_equal
import scipy.sparse as sp

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from ..pairs import PairTransformer
from ..pairs import CosineSimilarity
from ..pairs import AbsoluteDifference


def test_pair_transformer():
    """Test for PairTransformer."""
    X = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float)
    Xt = PairTransformer(element_transformer=MinMaxScaler()).fit_transform(X)
    assert_array_almost_equal(Xt, [[0, 0.2], [0.4, 0.6], [0.8, 1.0]])

    X = np.array([[0, 1], [2, 3]], dtype=np.float)
    tf = PairTransformer(element_transformer=OneHotEncoder(sparse=True))
    Xt = tf.fit_transform(X)
    assert sp.issparse(Xt)
    assert_array_almost_equal(Xt.todense(), [[1, 0, 0, 0, 0, 1, 0, 0],
                                             [0, 0, 1, 0, 0, 0, 0, 1]])

    X = sp.csr_matrix(np.array([[0, 1], [2, 3]], dtype=np.float))
    tf = PairTransformer(element_transformer=StandardScaler(with_mean=False))
    Xt = tf.fit_transform(X)
    assert sp.issparse(Xt)
    assert_array_almost_equal(Xt.todense(), [[0, 0.89442719],
                                             [1.78885438, 2.68328157]])


def test_cosine_similarity():
    """Test for CosineSimilarity."""
    X = np.array([[1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1]])

    Xt = CosineSimilarity().fit_transform(X)
    assert_array_almost_equal(Xt, [[0.], [2 ** -0.5], [1.], [0.], [1.]])

    Xt = CosineSimilarity().fit_transform(sp.csr_matrix(X))
    assert_array_almost_equal(Xt, [[0.], [2 ** -0.5], [1.], [0.], [1.]])


def test_absolute_difference():
    """Test for AbsoluteDifference."""
    X = np.array([[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [1, 1, 1, 1],
                  [1, 0, 0, 1]])

    Xt = AbsoluteDifference().fit_transform(X)
    assert_array_almost_equal(Xt, [[0, 0], [1, 1], [0, 0], [1, 1]])

    Xt = AbsoluteDifference().fit_transform(sp.csr_matrix(X))
    assert_array_almost_equal(Xt, [[0, 0], [1, 1], [0, 0], [1, 1]])
