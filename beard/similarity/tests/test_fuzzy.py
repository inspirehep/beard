# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Fuzzy Similarity for paired data.

.. codeauthor:: Evangelos Tzemis <evangelos.tzemis@cern.ch>

"""

from ..fuzzy import FuzzySimilarity
from beard.metrics import levenshtein
from numpy.testing import assert_almost_equal

import numpy as np


def test_fuzzy_simple():
    """Simple test for Fuzzy Similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Create a basic infrastructure for tests
    X = ["Je suis Evangelos",
         "Je suis Evangelo"]

    vect = TfidfVectorizer(min_df=1)
    fit = vect.fit(X)
    tfidf = vect.fit_transform(X)

    # Set vocabulary
    vocabulary = {k: v for v, k in fit.vocabulary_.items()}
    # Set distance function
    dist_f = levenshtein
    # Set threshold
    thrs = None
    # Set array to be transformed
    arr = tfidf.toarray()
    X = np.array([[item for sublist in arr for item in sublist]])

    FuzzySimilarity(vocabulary, dist_f, thrs).fit_transform(X)

    precomputed = arr[0, 1] * arr[1, 0]
    toCompare = FuzzySimilarity(vocabulary, dist_f, thrs).fit_transform(X)

    assert toCompare == precomputed


def test_fuzzy_complex():
    """Complex test for Fuzzy Similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Create a basic infrastructure for tests
    X = ["He runs fast",
         "She runs fast",
         "He runs faster"]

    vect = TfidfVectorizer(min_df=1)
    fit = vect.fit(X)
    tfidf = vect.fit_transform(X)

    # Set vocaBulary
    vocabulary = {k: v for v, k in fit.vocabulary_.items()}
    # Set distance function
    dist_f = levenshtein
    # Set threshold
    thrs = None
    # Set array to be transformed
    arr = tfidf.toarray()
    X = np.array([np.concatenate((arr[0], arr[1])).tolist(),
                  np.concatenate((arr[0], arr[2])).tolist(),
                  np.concatenate((arr[1], arr[2])).tolist()])

    FuzzySimilarity(vocabulary, dist_f, thrs).fit_transform(X)
    precomputed = (arr[0, 2]*arr[1, 4] +
                   arr[0, 0]*arr[2, 1]*2 +
                   arr[1, 4]*arr[2, 2] +
                   arr[1, 0]*arr[2, 1]*2)
    toCompare = FuzzySimilarity(vocabulary, dist_f, thrs).fit_transform(X)

    assert toCompare == precomputed


def test_find_distance():
    """Test _find_distance functionality."""
    document = ['You', 'are', 'him']
    word = 'her'
    vocabulary = {0: 'You', 1: 'are', 2: 'him', 3: 'her'}
    tfidf = np.array([1, 0.5, 1, 0])
    fuz = FuzzySimilarity(vocabulary, levenshtein, None)

    assert fuz._find_distance(word, tfidf) == (2, 2)
