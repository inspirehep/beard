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

Usage:

    1) Train a distance model

    python author_disambiguation_advanced.py \
        --distance_pairs=pairs.json \
        --distance_model=model.dat \
        --input_signatures=signatures.json \
        --input_records=records.json

    2) Cluster signatures using a trained distance model

    python author_disambiguation_advanced.py \
        --distance_model=model.dat \
        --input_signatures=signatures.json \
        --input_records=records.json \
        --output_clusters=clusters.json

    3) Evaluate disambiguation on known clusters

    python author_disambiguation_advanced.py \
        --distance_model=model.dat \
        --input_signatures=signatures.json \
        --input_records=records.json \
        --input_clusters=clusters.json \
        --clustering_test_size=0.9 \
        --verbose=1

Input files are expected to be formatted in JSON, using the following
conventions:

    - signatures.json : list of dictionaries holding metadata about signatures

        [{"signature_id": 0,
          "author_name": "Doe, John",
          "publication_id": 10, ...}, { ... }, ...]

    - records.json : list of dictionaries holding metadata about records

        [{"publication_id": 0,
          "title": "Author disambiguation using Beard", ... }, { ... }, ...]

    - clusters.json : dictionary, where keys are cluster labels and values
        are the `signature_id` of the signatures grouped in the clusters.
        Signatures assigned to the cluster with label "-1" are not clustered.

        {"0": [0, 1, 3], "1": [2, 5], ...}

        Note: predicted clusters are output in the same format.

    - pairs.json : list of tuples (`signature_id1`, `signature_id2`, `target`),
        where `target = 0` if both signatures belong to the same author,
        and `target = 1` otherwise.

        [(0, 1, 0), (2, 3, 0), (4, 5, 1), ...]

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

from __future__ import print_function

import argparse
import json
import pickle
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import squareform

from beard.clustering import BlockClustering
from beard.clustering import ScipyHierarchicalClustering
from beard.metrics import b3_f_score
from beard.similarity import PairTransformer
from beard.similarity import CosineSimilarity
from beard.similarity import AbsoluteDifference
from beard.utils import normalize_name
from beard.utils import name_initials
from beard.utils import FuncTransformer
from beard.utils import Shaper


def load_signatures(signatures_filename, records_filename):
    signatures = json.load(open(signatures_filename, "r"))
    records = json.load(open(records_filename, "r"))

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
    # Parse command line arugments
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance_pairs", default=None, type=str)
    parser.add_argument("--distance_model", default=None, type=str)
    parser.add_argument("--input_signatures", default=None, type=str)
    parser.add_argument("--input_records", default=None, type=str)
    parser.add_argument("--input_clusters", default=None, type=str)
    parser.add_argument("--output_clusters", default=None, type=str)
    parser.add_argument("--clustering_method", default="average", type=str)
    parser.add_argument("--clustering_threshold", default=None, type=float)
    parser.add_argument("--clustering_test_size", default=None, type=float)
    parser.add_argument("--clustering_random_state", default=42, type=int)
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("--n_jobs", default=-1, type=int)
    args = parser.parse_args()

    # Learn a distance model
    if args.distance_pairs:
        pairs = json.load(open(args.distance_pairs, "r"))
        signatures, records = load_signatures(args.input_signatures,
                                              args.input_records)

        X = np.empty((len(pairs), 2), dtype=np.object)
        y = np.empty(len(pairs), dtype=np.int)

        for k, (i, j, target) in enumerate(pairs):
            X[k, 0] = signatures[i]
            X[k, 1] = signatures[j]
            y[k] = target

        # Learn a distance estimator on paired signatures
        distance_estimator = build_distance_estimator(X, y)
        pickle.dump(distance_estimator,
                    open(args.distance_model, "w"),
                    protocol=pickle.HIGHEST_PROTOCOL)

    # Clustering
    else:
        distance_estimator = pickle.load(open(args.distance_model, "r"))
        signatures, records = load_signatures(args.input_signatures,
                                              args.input_records)

        indices = {}
        X = np.empty((len(signatures), 1), dtype=np.object)
        for i, signature in enumerate(signatures.values()):
            X[i, 0] = signature
            indices[signature["signature_id"]] = i

        # Semi-supervised block clustering
        if args.input_clusters:
            true_clusters = json.load(open(args.input_clusters, "r"))
            y_true = -np.ones(len(X), dtype=np.int)

            for label, signature_ids in true_clusters.items():
                for signature_id in signature_ids:
                    y_true[indices[signature_id]] = label

            if args.clustering_test_size is not None:
                train, test = train_test_split(
                    np.arange(len(X)),
                    test_size=args.clustering_test_size,
                    random_state=args.clustering_random_state)

                y = -np.ones(len(X), dtype=np.int)
                y[train] = y_true[train]

            else:
                y = y_true

        else:
            y = None

        clusterer = BlockClustering(
            blocking=blocking,
            base_estimator=ScipyHierarchicalClustering(
                affinity=affinity,
                threshold=args.clustering_threshold,
                method=args.clustering_method,
                scoring=b3_f_score),
            verbose=args.verbose,
            n_jobs=args.n_jobs).fit(X, y)

        labels = clusterer.labels_

        # Save predicted clusters
        if args.output_clusters:
            clusters = {}

            for label in np.unique(labels):
                mask = (labels == label)
                clusters[label] = [r[0]["signature_id"] for r in X[mask]]

            json.dump(clusters, open(args.output_clusters, "w"))

        # Statistics
        if args.verbose and args.input_clusters:
            print("Number of blocks =", len(clusterer.clusterers_))
            print("True number of clusters", len(np.unique(y_true)))
            print("Number of computed clusters", len(np.unique(labels)))
            print("B^3 F-score (overall) =", b3_f_score(y_true, labels))

            if args.clustering_test_size:
                print("B^3 F-score (train) =",
                      b3_f_score(y_true[train], labels[train]))
                print("B^3 F-score (test) =",
                      b3_f_score(y_true[test], labels[test]))
