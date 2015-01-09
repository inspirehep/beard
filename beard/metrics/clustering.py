# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Clustering evaluation metrics.

.. codeauthor:: Evangelos Tzemis <evangelos.tzemis@cern.ch>

"""
from __future__ import division
import numpy as np
from itertools import groupby
from sklearn.metrics.cluster.supervised import check_clusterings


def paired_precision_recall_fscore(labels_true, labels_pred):
    """Compute the pairwise variant of precision, recall and F-score.

    Precision is the ability not to label as positive a sample
    that is negative. The best value is 1 and the worst is 0.

    Recall is the ability to succesfully find all the positive samples.
    The best value is 1 and the worst is 0.

    F-score (Harmonic mean) can be thought as a weighted harmonic mean of
    the precision and recall, where an F-score reaches its best value at 1
    and worst at 0.

    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precission
    :return float recall: calculated recall
    :return float harmonic_mean: calculated harmonic_mean

    Reference
    ---------
    Levin, Michael et al., "Citation-based bootstrapping for large-scale
    author disambiguation", Journal of the American Society for Information
    Science and Technology 63.5 (2012): 1030-1047.

    """
    # Check that labels_* are 1d arrays and have the same size
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # Check that input given is not the empty set
    if labels_true.shape == (0, ):
        raise ValueError(
            "input labels must not be empty.")

    # Reconstruct the given input by grouping the samples
    c_gold = _group_samples_by_cluster_id(labels_true)
    c_system = _group_samples_by_cluster_id(labels_pred)

    # Calculate pairs (list of tuples)
    c_gold_pairs = _calculate_pairs(c_gold)
    c_system_pairs = _calculate_pairs(c_system)

    # Calculate evaluation metrics
    inters = c_gold_pairs.intersection(c_system_pairs)

    precision = len(inters)/len(c_system_pairs)
    recall = len(inters)/len(c_gold_pairs)
    # (precision+recall) always > 0
    harmonic_mean = 2*precision*recall/(precision + recall)

    return precision, recall, harmonic_mean


def paired_precision_score(labels_true, labels_pred):
    """Compute the pairwise variant of precision.

    Precision is the ability not to label as positive a sample
    that is negative. The best value is 1 and the worst is 0.

    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.

    Returns
    -------
    :return float precision: calculated precission

    """
    p, _, _ = paired_precision_recall_fscore(labels_true, labels_pred)
    return p


def paired_recall_score(labels_true, labels_pred):
    """Compute the pairwise variant of recall.

    Recall is the ability to succesfully find all the positive samples.
    The best value is 1 and the worst is 0.

    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth labels.
    :param labels_pred: 1d array containing the predicted labels.

    Returns
    -------
    :return float recall: calculated recall

    """
    _, r, _ = paired_precision_recall_fscore(labels_true, labels_pred)
    return r


def paired_f_score(labels_true, labels_pred):
    """Compute the pairwise variant of F score.

    F score can be thought as a weighted harmonic mean of the precision
    and recall, where an F score reaches its best value at 1
    and worst at 0.

    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.

    Returns
    -------
    :return float harmonic_mean: calculated harmonic mean (f_score)

    """
    _, _, f = paired_precision_recall_fscore(labels_true, labels_pred)
    return f


def _calculate_pairs(labels):
    """Find all possible pair combination for given input.

    Parameters
    ----------
    :param labels: array of lists containing the ids of elements of cluster.

    Returns
    -------
    :return: all possible pair combinations

    """
    pairs = [(i, j) for cluster in labels for i in cluster for j in cluster]

    # Remove dublicated pairs: (x,y) equals to (y,x)
    pairs_set = set(map(frozenset, pairs))

    return pairs_set


def _group_samples_by_cluster_id(labels):
    """Group input to sets that belong to the same cluster.

    Parameters
    ----------
    :param labels: array with the cluster labels

    Returns
    -------
    :return: generator of lists containing the ids of elements of cluster.

    """
    groupped_samples = groupby(np.argsort(labels), lambda i: labels[i])

    return (list(group) for _, group in groupped_samples)
