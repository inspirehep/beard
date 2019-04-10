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
.. codeauthor:: Hussein AL-NATSHEH <hussein.al.natsheh@cern.ch>

"""

import jellyfish
import numpy as np
from numpy.testing import assert_array_almost_equal
import scipy.sparse as sp

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.cross_validation import train_test_split
except ImportError:
    from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC

from beard.similarity import AbsoluteDifference
from beard.similarity import CosineSimilarity
from beard.similarity import ElementMultiplication
from beard.similarity import EstimatorTransformer
from beard.similarity import JaccardSimilarity
from beard.similarity import PairTransformer
from beard.similarity import StringDistance
from beard.utils import FuncTransformer


def test_pair_transformer():
    """Test for PairTransformer."""
    X = np.array([[0, 1], [2, 0], [2, 5]], dtype=np.float)
    tf = PairTransformer(element_transformer=FuncTransformer(lambda v: v + 1))
    Xt = tf.fit_transform(X)
    assert_array_almost_equal(Xt, X + 1)

    X = np.array([[0, 1], [2, 0], [2, 5],
                  [0, 1], [2, 0], [2, 5]], dtype=np.float)
    tf = PairTransformer(element_transformer=FuncTransformer(lambda v: v + 1),
                         groupby=lambda r: r[0])
    Xt = tf.fit_transform(X)
    assert_array_almost_equal(Xt, X + 1)

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


def test_CharacterEquality():
    """Test for CharacterEquality."""
    X = np.array([['q', 'q'],
                  ['q', 'a'],
                  ['q', ''],
                  ['', ''],
                  ['', 'q']])
    Xt = StringDistance(similarity_function='character_equality').transform(X)
    assert_array_almost_equal(Xt, [[1.], [0.], [0.], [0.5], [0.]])


def test_StringDistance():
    """Test for StringDistance."""
    X = np.array([[u'this', u'that'],
                  [u'that', u't'],
                  [u't', u't'],
                  [u't', u'this']])
    Xt = StringDistance().transform(X)
    assert_array_almost_equal(Xt, [[jellyfish.jaro_winkler(u'this', u'that')],
                                   [-1.], [-1.], [-1.]])


def test_JaccardSimilarity():
    """Test for JaccardSimilarity."""
    X = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 1, 0, 1],
                  [0, 1, 0, 1, 0, 0, 1, 0],
                  [1, 0, 1, 1, 1, 1, 0, 7],
                  [0, 3, 0, 1, 0, 9, 0, 1]])

    Xt = JaccardSimilarity().fit_transform(X)
    assert_array_almost_equal(Xt, [[0.], [0.33333333], [0.], [0.5], [1.]])

    Xt = JaccardSimilarity().fit_transform(sp.csr_matrix(X))
    assert_array_almost_equal(Xt, [[0.], [0.33333333], [0.], [0.5], [1.]])

    X = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])

    Xt = JaccardSimilarity().fit_transform(X)
    assert_array_almost_equal(Xt, [[0.], [0.], [0.], [0.]])

    Xt = JaccardSimilarity().fit_transform(sp.csr_matrix(X))
    assert_array_almost_equal(Xt, [[0.], [0.], [0.], [0.]])


def test_EstimatorTransformer():
    """Test for EstimatorTransformer."""
    data = load_iris()
    train, test = train_test_split(np.arange(len(data.data)),
                                   test_size=0.08, random_state=42)
    X_train = data.data[train]
    y_train = data.target[train]
    X_test = data.data[test]

    clf = LinearSVC().fit(X_train, y_train)

    y_predict = clf.decision_function(X_test)

    Xt = EstimatorTransformer(clf).fit_transform(X_test)
    assert_array_almost_equal(Xt, y_predict)


def test_ElementMultiplication():
    """Test for ElementMultiplication."""
    X = np.array([[1.0, 1.0, 1.0, 2.0],
                  [0.5, 1.0, 1.0, 0.5],
                  [2.5, 0.2, 10.0, 2.0]])

    y = np.array([[1.0, 2.0],
                  [0.5, 0.5],
                  [25.0, 0.4]])

    Xt = ElementMultiplication().fit_transform(X)
    assert_array_almost_equal(Xt, y)
