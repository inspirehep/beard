# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Simplified author disambiguation example.

This example shows how to use semi-supervised block clustering for the author
disambiguation problem. To goal is to cluster together all (author name,
affiliation) tuples that correspond to the same actual person.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

import numpy as np
import re

from functools import wraps
from sklearn.cross_validation import train_test_split

from beard.clustering import BlockClustering
from beard.clustering import ScipyHierarchicalClustering
from beard.metrics import paired_f_score
from beard.utils.strings import asciify


def memoize(func):
    """Memoization function."""
    cache = {}

    @wraps(func)
    def wrap(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrap


RE_NORMALIZE_LAST_NAME = re.compile("\s+|\-")
RE_NORMALIZE = re.compile("(,\s(i|ii|iii|iv|v|vi|jr))|[\.'\-,]|\s+")


@memoize
def normalize(name):
    """Transliterate a name to ascii and remove all special characters."""
    name = asciify(name).lower()

    try:
        names = name.split(",", 1)
        name = "%s, %s" % (RE_NORMALIZE_LAST_NAME.sub("", names[0]), names[1])
    except:
        pass

    name = RE_NORMALIZE.sub(" ", name)
    name = name.strip()

    return name


@memoize
def initials(name):
    """Compute the set of initials of a given name."""
    return set([w[0] for w in name.split()])


def affinity(X):
    """Compute pairwise distances between (author, affiliation) tuples.

    Note that this function is a heuristic. It should ideally be replaced
    by a more robust distance function, e.g. using a model learned over
    pairs of tuples.
    """
    distances = np.zeros((len(X), len(X)), dtype=np.float)

    for i, j in zip(*np.triu_indices(len(X), k=1)):
        name_i = normalize(X[i, 0])
        aff_i = X[i, 1]
        initials_i = initials(name_i)
        name_j = normalize(X[j, 0])
        aff_j = X[j, 1]
        initials_j = initials(name_j)

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


def blocking(X):
    """Blocking function using last name and first initial as key."""
    def last_name_first_initial(name):
        try:
            last_name, other_names = name.split(",", 1)
            return "%s %s" % (last_name, other_names.strip()[0])
        except:
            return name

    blocks = []

    for name in X[:, 0]:
        blocks.append(normalize(last_name_first_initial(name)))

    return np.array(blocks)


if __name__ == "__main__":
    # Load data
    X = np.load("data/authors-X.npy")
    truth = np.load("data/authors-clusters.npy")

    # Split into train and test sets
    train, test = train_test_split(np.arange(len(X)),
                                   test_size=0.75, random_state=42)
    y = -np.ones(len(X), dtype=np.int)
    y[train] = truth[train]

    # Semi-supervised block clustering
    block_clusterer = BlockClustering(
        blocking=blocking,
        base_estimator=ScipyHierarchicalClustering(affinity=affinity,
                                                   method="complete"),
        n_jobs=-1)
    block_clusterer.fit(X, y=y)
    labels = block_clusterer.labels_

    # Print clusters
    for cluster in np.unique(labels):
        entries = set()

        for name, affiliation in X[labels == cluster]:
            entries.add((name, affiliation))

        print "Cluster #%d = %s" % (cluster, entries)
    print

    # Statistics
    print "Number of blocks =", len(block_clusterer.clusterers_)
    print "True number of clusters", len(np.unique(truth))
    print "Number of computed clusters", len(np.unique(labels))
    print "Paired F-score =", paired_f_score(truth, labels)
