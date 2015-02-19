# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Advanced author disambiguation example.

This example shows how to build a full author disambiguation pipeline.
The pipeline is made of two steps:

    1) Supervised learning, for inferring a distance or affinity function
       between publications. This estimator is learned from labeled paired data
       and models whether two publications have been authored by the same
       person.

    2) Semi-supervised block clustering, for grouping together publications
       from the same author. Publications are blocked by last name + first
       initial, and then clustered using hierarchical clustering together with
       the affinity function learned at the previous step. For each block,
       the best cut-off threshold is chosen so as to maximize some scoring
       metric on the provided labeled data.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

from __future__ import print_function

import pickle
import numpy as np
import sys

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import squareform

from beard.clustering import BlockClustering
from beard.clustering import ScipyHierarchicalClustering
from beard.metrics import paired_f_score
from beard.similarity import PairTransformer
from beard.similarity import CosineSimilarity
from beard.similarity import AbsoluteDifference
from beard.utils import normalize_name
from beard.utils import name_initials
from beard.utils import FuncTransformer
from beard.utils import Shaper


def resolve_publications(signatures, records):
    """Resolve the 'publication' field in signatures."""
    if isinstance(signatures, list):
        signatures = {s["signature_id"]: s for s in signatures}

    if isinstance(records, list):
        records = {r["publication_id"]: r for r in records}

    for signature_id, signature in signatures.items():
        signature["publication"] = records[signature["publication_id"]]

    return signatures, records


def get_author_full_name(s):
    v = s["author_name"]
    v = normalize_name(v) if v else ""
    return v


def get_author_other_names(s):
    v = s["author_name"]
    v = v.split(",", 1)
    v = normalize_name(v[1]) if len(v) == 2 else ""
    return v


def get_author_initials(s):
    v = s["author_name"]
    v = v if v else ""
    v = "".join(name_initials(v))
    return v


def get_author_affiliation(s):
    v = s["author_affiliation"]
    v = normalize_name(v) if v else ""
    return v


def get_title(s):
    v = s["publication"]["title"]
    v = v if v else ""
    return v


def get_journal(s):
    v = s["publication"]["journal"]
    v = v if v else ""
    return v


def get_abstract(s):
    v = s["publication"]["abstract"]
    v = v if v else ""
    return v


def get_coauthors(s):
    v = s["publication"]["authors"]
    v = " ".join(v)
    return v


def get_keywords(s):
    v = s["publication"]["keywords"]
    v = " ".join(v)
    return v


def get_collaborations(s):
    v = s["publication"]["collaborations"]
    v = " ".join(v)
    return v


def get_references(s):
    v = s["publication"]["references"]
    v = " ".join(str(r) for r in v)
    v = v if v else ""
    return v


def get_year(s):
    v = s["publication"]["year"]
    v = int(v) if v else -1
    return v


def group_by_signature(r):
    return r[0]["signature_id"]


def build_distance_estimator(X, y):
    # Build a vector reprensation of a pair of signatures
    transformer = FeatureUnion([
        ("author_full_name_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("full_name", FuncTransformer(func=get_author_full_name)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                           ngram_range=(2, 4),
                                           dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=group_by_signature)),
            ("combiner", CosineSimilarity())
        ])),
        ("author_other_names_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("other_names", FuncTransformer(func=get_author_other_names)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                           ngram_range=(2, 4),
                                           dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=group_by_signature)),
            ("combiner", CosineSimilarity())
        ])),
        ("author_initials_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("initials", FuncTransformer(func=get_author_initials)),
                ("shaper", Shaper(newshape=(-1,))),
                ("count", CountVectorizer(analyzer="char_wb",
                                          ngram_range=(1, 1),
                                          binary=True,
                                          decode_error="replace")),
            ]), groupby=group_by_signature)),
            ("combiner", CosineSimilarity())
        ])),
        ("affiliation_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("affiliation", FuncTransformer(func=get_author_affiliation)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                           ngram_range=(2, 4),
                                           dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=group_by_signature)),
            ("combiner", CosineSimilarity())
        ])),
        ("coauthors_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("coauthors", FuncTransformer(func=get_coauthors)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=group_by_signature)),
            ("combiner", CosineSimilarity())
        ])),
        ("title_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("title", FuncTransformer(func=get_title)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                           ngram_range=(2, 4),
                                           dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=group_by_signature)),
            ("combiner", CosineSimilarity())
        ])),
        ("journal_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("journal", FuncTransformer(func=get_journal)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                           ngram_range=(2, 4),
                                           dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=group_by_signature)),
            ("combiner", CosineSimilarity())
        ])),
        ("abstract_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("abstract", FuncTransformer(func=get_abstract)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=group_by_signature)),
            ("combiner", CosineSimilarity())
        ])),
        ("keywords_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("keywords", FuncTransformer(func=get_keywords)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=group_by_signature)),
            ("combiner", CosineSimilarity())
        ])),
        ("collaborations_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("collaborations", FuncTransformer(func=get_collaborations)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=group_by_signature)),
            ("combiner", CosineSimilarity())
        ])),
        ("references_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("references", FuncTransformer(func=get_references)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=group_by_signature)),
            ("combiner", CosineSimilarity())
        ])),
        ("year_diff", Pipeline([
            ("pairs", FuncTransformer(func=get_year, dtype=np.int)),
            ("combiner", AbsoluteDifference())  # FIXME: when one is missing
        ]))])

    # Train a classifier on these vectors
    classifier = GradientBoostingClassifier(n_estimators=500,
                                            max_depth=9,
                                            max_features=10,
                                            learning_rate=0.125,
                                            verbose=3)

    # Return the whole pipeline
    estimator = Pipeline([("transformer", transformer),
                          ("classifier", classifier)]).fit(X, y)

    return estimator


def affinity(X, step=10000):
    """Custom affinity function, using a pre-learned distance estimator."""
    # This assumes that 'distance_estimator' lives in global

    all_i, all_j = np.triu_indices(len(X), k=1)
    n_pairs = len(all_i)
    distances = np.zeros(n_pairs)

    for start in range(0, n_pairs, step):
        end = min(n_pairs, start+step)
        Xt = np.empty((end-start, 2), dtype=np.object)

        for k, (i, j) in enumerate(zip(all_i[start:end],
                                       all_j[start:end])):
            Xt[k, 0], Xt[k, 1] = X[i, 0], X[j, 0]

        Xt = distance_estimator.predict_proba(Xt)[:, 1]
        distances[start:end] = Xt[:]

    return squareform(distances)


def blocking(X):
    """Blocking function using last name and first initial as key."""
    def last_name_first_initial(name):
        names = name.split(",", 1)

        try:
            name = "%s %s" % (names[0], names[1].strip()[0])
        except IndexError:
            name = names[0]

        name = normalize_name(name)
        return name

    blocks = []

    for signature in X[:, 0]:
        blocks.append(last_name_first_initial(signature["author_name"]))

    return np.array(blocks)


if __name__ == "__main__":
    # Load paired data
    X, y, signatures, records = pickle.load(open(sys.argv[1], "r"))
    signatures, records = resolve_publications(signatures, records)

    Xt = np.empty((len(X), 2), dtype=np.object)
    for k, (i, j) in enumerate(X):
        Xt[k, 0] = signatures[i]
        Xt[k, 1] = signatures[j]
    X = Xt

    # Learn a distance estimator on paired signatures
    distance_estimator = build_distance_estimator(X, y)

    # Load signatures to cluster
    signatures, truth, records = pickle.load(open(sys.argv[2], "r"))
    _, records = resolve_publications(signatures, records)

    X = np.empty((len(signatures), 1), dtype=np.object)
    for i, signature in enumerate(signatures):
        X[i, 0] = signature

    # Semi-supervised block clustering
    train, test = train_test_split(np.arange(len(X)),
                                   test_size=0.75, random_state=42)
    y = -np.ones(len(X), dtype=np.int)
    y[train] = truth[train]

    clusterer = BlockClustering(
        blocking=blocking,
        base_estimator=ScipyHierarchicalClustering(
            threshold=0.9995,
            affinity=affinity,
            method="complete"),
        verbose=3,
        n_jobs=-1).fit(X, y)

    labels = clusterer.labels_

    # Print clusters
    for cluster in np.unique(labels):
        entries = set()

        for signature in X[labels == cluster, 0]:
            entries.add((signature["author_name"],
                         signature["author_affiliation"]))

        print("Cluster #%d = %s" % (cluster, entries))
    print()

    # Statistics
    print("Number of blocks =", len(clusterer.clusterers_))
    print("True number of clusters", len(np.unique(truth)))
    print("Number of computed clusters", len(np.unique(labels)))
    print("Paired F-score (overall) =", paired_f_score(truth, labels))
    print("Paired F-score (train) =", paired_f_score(truth[train],
                                                     labels[train]))
    print("Paired F-score (test) =", paired_f_score(truth[test], labels[test]))
