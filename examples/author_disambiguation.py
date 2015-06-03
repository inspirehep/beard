# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Simplified author disambiguation example.

This example shows how to use block clustering for the author
disambiguation problem. To goal is to cluster together all (author name,
affiliation) tuples that correspond to the same actual person.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

from __future__ import print_function

import numpy as np

from beard.clustering import BlockClustering
from beard.clustering import block_last_name_first_initial
from beard.clustering import ScipyHierarchicalClustering
from beard.metrics import paired_f_score
from beard.utils import normalize_name
from beard.utils import name_initials


def affinity(X):
    """Compute pairwise distances between (author, affiliation) tuples.

    Note that this function is a heuristic. It should ideally be replaced
    by a more robust distance function, e.g. using a model learned over
    pairs of tuples.
    """
    distances = np.zeros((len(X), len(X)), dtype=np.float)

    for i, j in zip(*np.triu_indices(len(X), k=1)):
        name_i = normalize_name(X[i, 0])
        aff_i = X[i, 1]
        initials_i = name_initials(name_i)
        name_j = normalize_name(X[j, 0])
        aff_j = X[j, 1]
        initials_j = name_initials(name_j)

        # Names and affiliations match
        if (name_i == name_j and aff_i == aff_j):
            distances[i, j] = 0.0

        # Compatible initials and affiliations match
        elif (len(initials_i | initials_j) == max(len(initials_i),
                                                  len(initials_j)) and
              aff_i == aff_j and aff_i != ""):
            distances[i, j] = 0.0

        # Initials are not compatible
        elif (len(initials_i | initials_j) != max(len(initials_i),
                                                  len(initials_j))):
            distances[i, j] = 1.0

        # We dont know
        else:
            distances[i, j] = 0.5

    distances += distances.T
    return distances

if __name__ == "__main__":
    # Load data
    data = np.load("data/author-disambiguation.npz")
    X = data["X"]
    truth = data["y"]

    # Block clustering with fixed threshold
    block_clusterer = BlockClustering(
        blocking=block_last_name_first_initial,
        base_estimator=ScipyHierarchicalClustering(
            threshold=0.5,
            affinity=affinity,
            method="complete"),
        verbose=3,
        n_jobs=-1)
    block_clusterer.fit(X)
    labels = block_clusterer.labels_

    # Print clusters
    for cluster in np.unique(labels):
        entries = set()

        for name, affiliation in X[labels == cluster]:
            entries.add((name, affiliation))

        print("Cluster #%d = %s" % (cluster, entries))
    print()

    # Statistics
    print("Number of blocks =", len(block_clusterer.clusterers_))
    print("True number of clusters", len(np.unique(truth)))
    print("Number of computed clusters", len(np.unique(labels)))
    print("Paired F-score =", paired_f_score(truth, labels))
