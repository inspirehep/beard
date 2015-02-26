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

import numpy as np
import scipy.sparse as sp
import operator

from sklearn.base import TransformerMixin


class FuzzySimilarity(TransformerMixin):

    """Implements Fuzzy Similarity (Soft TF-IDF) metric."""

    def __init__(self, vocabulary, distance, threshold):
        """Initialize.

        Parameters
        ----------
        :param vocabulary:
            From word to feature_id or opposite.

        :param distance:
            Distance function which returns anumper in R.

        :param threshold:
            Threshold parameter that belongs to R.
        """
        self.vocabulary = vocabulary
        self.distance = distance
        self.threshold = threshold

    def fit(self, X, y=None):
        """Return self."""
        return self

    def transform(self, X):
        """Compute the fuzzy similarity for all pairs of elements in `X`.

        Rows i in `X` are assumed to represent pairs, where
        `X[i, :n_features]` and `X[i, n_features:]` correspond to their two
        individual elements. Calling `transform` computes the fuzzy similarity
        between between these elements, i.e. that `X_[i]` is the fuzzy of
        `X[i, :n_features]` and `X[i, n_features:]`.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2 * n_features)
            Input paired data.

        Returns
        -------
        :returns X_: array-like, shape (n_samples, 2 * n_features_prime)
            The transformed data.
        """
        n_samples, n_features_all = X.shape
        n_features = n_features_all / 2

        if sp.issparse(X):
            X = X.todense()

        X1 = X[:, :n_features]
        X2 = X[:, n_features:]

        summ = 0
        # for each document in X1
        for row, u in enumerate(X1):
            # for each word in document u
            for pos, w in enumerate(u):
                if w > 0.0:
                    w_word = self.vocabulary[pos]
                    min_d, n_index = self._find_distance(w_word, X2[row, :])
                    summ += w * X2[row, n_index] * min_d
        return summ

    def _find_distance(self, u, words):
        """Find the distance between u and words according to threshold.

        If threshold is 'None' then we return the minimum distance between u
        and one of the words contained in 'words'. Otherwise we just return
        the first distance we find being less than the threshold.

        """
        positive_tuples = filter(lambda (i, x): x > 0.0, enumerate(words))
        words_indices = map(lambda (i, v): (i, self.vocabulary[i]),
                            positive_tuples)

        if self.threshold is None:
            index_distances = map(lambda (i, x): (i, self.distance(u, x)),
                                  words_indices)
            min_index, min_distance = min(index_distances,
                                          key=operator.itemgetter(1))
        else:   # TODO
            min_distance, min_index = 0, 0

        return min_distance, min_index
