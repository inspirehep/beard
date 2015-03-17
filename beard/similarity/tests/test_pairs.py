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
.. codeauthor:: Evangelos Tzemis <evangelos.tzemis@cern.ch>

"""

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
import scipy.sparse as sp

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from ..pairs import PairTransformer
from ..pairs import CosineSimilarity
from ..pairs import AbsoluteDifference
from ..pairs import FuzzySimilarity
from ...utils import FuncTransformer

from beard.metrics import jaro
from beard.metrics import jaro_winkler


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


def test_fuzzy_simple():
    """Simple test for Fuzzy Similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = ["Je suis Evangelos",
         "Je suis Evangelo"]

    vect = TfidfVectorizer(min_df=1)
    fit = vect.fit(X)
    tfidf = vect.fit_transform(X)

    # Set parameters
    vocabulary = {k: v for v, k in fit.vocabulary_.items()}
    sim_f = jaro
    thrs = None
    arr = tfidf.toarray()
    X = np.array([[item for sublist in arr for item in sublist]])

    # Compute the expected result
    Pre_X1 = np.array(arr[0]).reshape((1, 4))
    Pre_X1_closest = np.array([Pre_X1[0, 1], Pre_X1[0, 1],
                               Pre_X1[0, 2], Pre_X1[0, 3]]).reshape((1, 4))
    Pre_Similarities = np.array([1, jaro('Evangelos', 'Evangelo'),
                                 1, 1]).reshape((1, 4))

    numerator = (Pre_X1 * Pre_X1_closest * Pre_Similarities).sum(axis=1)
    norm1 = (Pre_X1 * Pre_X1).sum(axis=1)
    norm2 = ((Pre_X1_closest * Pre_Similarities)*
             (Pre_X1_closest * Pre_Similarities)).sum(axis=1)

    denominator = (norm1 ** 0.5) * (norm2 ** 0.5)

    with np.errstate(divide="ignore"):
        Xt = numerator / denominator
        Xt[denominator == 0.0] = 0.0

    # Produce and compare the values
    preComputed = Xt.reshape((1, 1))
    toCompare = FuzzySimilarity().transform(X, vocabulary, sim_f, thrs)

    assert_array_almost_equal(toCompare, preComputed)


def test_fuzzy_complex():
    """Complex test for Fuzzy Similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = ["apple lemon",
         "apple orange",
         "orange lemon"]

    vect = TfidfVectorizer(min_df=1)
    fit = vect.fit(X)
    tfidf = vect.fit_transform(X)

    # Set parameters
    vocabulary = {k: v for v, k in fit.vocabulary_.items()}
    sim_f = jaro
    thrs = None
    arr = tfidf.toarray()
    X = np.array([np.concatenate((arr[0], arr[1])).tolist(),
                  np.concatenate((arr[0], arr[2])).tolist(),
                  np.concatenate((arr[1], arr[2])).tolist()])

    # Compute the expected result
    sim_a_o = jaro('apple', 'orange')
    sim_o_l = jaro('orange', 'lemon')
    tf_common = arr[0,0]

    Pre_X1 = np.array([[tf_common, tf_common, 0],
                       [tf_common, tf_common, 0],
                       [tf_common, 0, tf_common]]).reshape((3, 3))
    Pre_X1_closest = np.array([[tf_common, tf_common, tf_common],
                               [tf_common, tf_common, tf_common],
                               [tf_common, tf_common, tf_common]]).reshape((3, 3))
    Pre_Similarities = np.array([[1, sim_o_l, 1],
                                 [sim_a_o, 1, 1],
                                 [sim_a_o, 1, 1]]).reshape((3,3))
    numerator = (Pre_X1 * Pre_X1_closest * Pre_Similarities).sum(axis=1)
    norm1 = (Pre_X1 * Pre_X1).sum(axis=1)
    norm2 = ((Pre_X1_closest * Pre_Similarities)*
             (Pre_X1_closest * Pre_Similarities)).sum(axis=1)

    denominator = (norm1 ** 0.5) * (norm2 ** 0.5)

    with np.errstate(divide="ignore"):
        Xt = numerator / denominator
        Xt[denominator == 0.0] = 0.0

    # Produce and compare the values
    preComputed = Xt.reshape((3, 1))
    toCompare = FuzzySimilarity().transform(X, vocabulary, sim_f, thrs)

    assert_array_almost_equal(toCompare, preComputed)


def test_fuzzy_threshold():
    """Test threshold for Fuzzy Similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = ["apple lemon",
         "apple orange",
         "orange lemon"]

    vect = TfidfVectorizer(min_df=1)
    fit = vect.fit(X)
    tfidf = vect.fit_transform(X)

    # Set parameters
    vocabulary = {k: v for v, k in fit.vocabulary_.items()}
    sim_f = jaro
    thrs = 1.0
    arr = tfidf.toarray()
    X = np.array([np.concatenate((arr[0], arr[1])).tolist(),
                  np.concatenate((arr[0], arr[2])).tolist(),
                  np.concatenate((arr[1], arr[2])).tolist()])

    # Compute the expected result
    sim_a_o = jaro('apple', 'orange')
    sim_o_l = jaro('orange', 'lemon')
    tf_common = arr[0,0]

    Pre_X1 = np.array([[tf_common, tf_common, 0],
                       [tf_common, tf_common, 0],
                       [tf_common, 0, tf_common]]).reshape((3, 3))
    Pre_X1_closest = np.array([[tf_common, tf_common, tf_common],
                               [tf_common, tf_common, tf_common],
                               [tf_common, tf_common, tf_common]]).reshape((3, 3))
    Pre_Similarities = np.array([[1, 0, 1],
                                 [0, 1, 1],
                                 [0, 1, 1]]).reshape((3,3))
    numerator = (Pre_X1 * Pre_X1_closest * Pre_Similarities).sum(axis=1)
    norm1 = (Pre_X1 * Pre_X1).sum(axis=1)
    norm2 = ((Pre_X1_closest * Pre_Similarities)*
             (Pre_X1_closest * Pre_Similarities)).sum(axis=1)

    denominator = (norm1 ** 0.5) * (norm2 ** 0.5)

    with np.errstate(divide="ignore"):
        Xt = numerator / denominator
        Xt[denominator == 0.0] = 0.01

    # Produce and compare the values
    preComputed = Xt.reshape((3, 1))
    toCompare = FuzzySimilarity().transform(X, vocabulary, sim_f, thrs)

    assert_array_almost_equal(toCompare, preComputed)


def test_fuzzy_induction_to_cosine():
    """ Test fuzzy similarity's ability to induct to cosine similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = ["Je suis Evangelos",
         "Je suis Evangelo"]

    vect = TfidfVectorizer(min_df=1)
    fit = vect.fit(X)
    tfidf = vect.fit_transform(X)

    # Set vocabulary
    vocabulary = {k: v for v, k in fit.vocabulary_.items()}

    # Set distance function
    def simple_similarity_function(w1, w2):
        # Returns 0 if strings are not identical, or 1 otherwise
        if w1 == w2:
            return 1
        return 0
    sim_f = simple_similarity_function
    
    # Set threshold
    thrs = 1
    # Set array to be transformed
    arr = tfidf.toarray()
    X = np.array([[item for sublist in arr for item in sublist]])

    Y = FuzzySimilarity().transform(X, vocabulary, sim_f, thrs)
    Y_prime = CosineSimilarity().fit_transform(X)

    assert_array_almost_equal(Y, Y_prime)


def test_fuzzy_sparse_matrix():
    """Sparse matrix support test for Fuzzy Similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = ["Je suis Evangelos",
         "Je suis Evangelo"]

    vect = TfidfVectorizer(min_df=1)
    fit = vect.fit(X)
    tfidf = vect.fit_transform(X)

    # Set parameters
    vocabulary = {k: v for v, k in fit.vocabulary_.items()}
    sim_f = jaro
    thrs = None
    arr = tfidf.toarray()
    X = np.array([[item for sublist in arr for item in sublist]])
    X = sp.csr_matrix(X)

    # Compute the expected result
    Pre_X1 = np.array(arr[0]).reshape((1, 4))
    Pre_X1_closest = np.array([Pre_X1[0, 1], Pre_X1[0, 1],
                               Pre_X1[0, 2], Pre_X1[0, 3]]).reshape((1, 4))
    Pre_Similarities = np.array([1, jaro('Evangelos', 'Evangelo'),
                                 1, 1]).reshape((1, 4))

    numerator = (Pre_X1 * Pre_X1_closest * Pre_Similarities).sum(axis=1)
    norm1 = (Pre_X1 * Pre_X1).sum(axis=1)
    norm2 = ((Pre_X1_closest * Pre_Similarities)*
             (Pre_X1_closest * Pre_Similarities)).sum(axis=1)

    denominator = (norm1 ** 0.5) * (norm2 ** 0.5)

    with np.errstate(divide="ignore"):
        Xt = numerator / denominator
        Xt[denominator == 0.0] = 0.0

    # Produce and compare the values
    preComputed = Xt.reshape((1, 1))
    toCompare = FuzzySimilarity().transform(X, vocabulary, sim_f, thrs)

    assert_array_almost_equal(toCompare, preComputed)
