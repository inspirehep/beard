# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Author disambiguation -- Clustering.

See README.rst for further details.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

import argparse
import pickle
import json
import numpy as np

from sklearn.cross_validation import train_test_split

# These imports are used during unpickling.
from utils import get_author_full_name
from utils import get_author_other_names
from utils import get_author_initials
from utils import get_author_affiliation
from utils import get_title
from utils import get_journal
from utils import get_abstract
from utils import get_coauthors_from_range
from utils import get_keywords
from utils import get_collaborations
from utils import get_references
from utils import get_topics
from utils import get_year
from utils import group_by_signature
from utils import load_signatures

from beard.clustering import BlockClustering
from beard.clustering import block_last_name_first_initial
from beard.clustering import ScipyHierarchicalClustering
from beard.metrics import b3_f_score
from beard.metrics import b3_precision_recall_fscore
from beard.metrics import paired_precision_recall_fscore


def _affinity(X, step=10000):
    """Custom affinity function, using a pre-learned distance estimator."""
    # Assumes that 'distance_estimator' lives in global, making things fast
    global distance_estimator

    all_i, all_j = np.triu_indices(len(X), k=1)
    n_pairs = len(all_i)
    distances = np.zeros(n_pairs, dtype=np.float64)

    for start in range(0, n_pairs, step):
        end = min(n_pairs, start+step)
        Xt = np.empty((end-start, 2), dtype=np.object)

        for k, (i, j) in enumerate(zip(all_i[start:end],
                                       all_j[start:end])):
            Xt[k, 0], Xt[k, 1] = X[i, 0], X[j, 0]

        Xt = distance_estimator.predict_proba(Xt)[:, 1]
        distances[start:end] = Xt[:]

    return distances


def clustering(input_signatures, input_records, distance_model,
               input_clusters=None, output_clusters=None,
               verbose=1, n_jobs=-1, clustering_method="average",
               train_signatures_file=None, clustering_threshold=None,
               results_file=None):
    """Cluster signatures using a pretrained distance model.

    Parameters
    ----------
    :param input_signatures: string
        Path to the file with signatures. The content should be a JSON array
        of dictionaries holding metadata about signatures.

        [{"signature_id": 0,
          "author_name": "Doe, John",
          "publication_id": 10, ...}, { ... }, ...]

    :param input_records: string
        Path to the file with records. The content should be a JSON array of
        dictionaries holding metadata about records

        [{"publication_id": 0,
          "title": "Author disambiguation using Beard", ... }, { ... }, ...]

    :param distance_model: string
        Path to the file with the distance model. The file should be a pickle
        created using the ``distance.py`` script.

    :param input_clusters: string
        Path to the file with knownn clusters. The file should be a dictionary,
        where keys are cluster labels and values are the `signature_id` of the
        signatures grouped in the clusters. Signatures assigned to the cluster
        with label "-1" are not clustered.

        {"0": [0, 1, 3], "1": [2, 5], ...}

    :param output_clusters: string
        Path to the file with output cluster. The file will be filled with
        clusters, using the same format as ``input_clusters``.

    :param verbose: int
        If not zero, function will output scores on stdout.

    :param n_jobs: int
        Parameter passed to joblib. Number of threads to be used.

    :param clustering_method: string
        Parameter passed to ``ScipyHierarchicalClustering``. Used only if
        ``clustering_test_size`` is specified.

    :param train_signatures_file: str
        Path to the file with train set signatures. Format the same as in
        ``input_signatures``.

    :param clustering_threshold: float
        Threshold passed to ``ScipyHierarchicalClustering``.

    :param results_file: str
        Path to the file where the results will be output. It will give
        additional information about pairwise variant of scores.
    """
    # Assumes that 'distance_estimator' lives in global, making things fast
    global distance_estimator

    distance_estimator = pickle.load(open(distance_model, "rb"))
    signatures, records = load_signatures(input_signatures,
                                          input_records)

    indices = {}
    X = np.empty((len(signatures), 1), dtype=np.object)
    for i, signature in enumerate(sorted(signatures.values(),
                                         key=lambda s: s["signature_id"])):
        X[i, 0] = signature
        indices[signature["signature_id"]] = i

    # Semi-supervised block clustering
    if input_clusters:
        true_clusters = json.load(open(input_clusters, "r"))
        y_true = -np.ones(len(X), dtype=np.int)

        for label, signature_ids in true_clusters.items():
            for signature_id in signature_ids:
                y_true[indices[signature_id]] = label

        y = -np.ones(len(X), dtype=np.int)

        if train_signatures_file:
            train_signatures = json.load(open(train_signatures_file, "r"))
            train_ids = [x['signature_id'] for x in train_signatures]
            del train_signatures
            y[train_ids] = y_true[train_ids]
            test_ids = list(set([x['signature_id'] for _, x in
                                 signatures.iteritems()]) - set(train_ids))
        else:
            y = y_true

    else:
        y = None

    clusterer = BlockClustering(
        blocking=block_last_name_first_initial,
        base_estimator=ScipyHierarchicalClustering(
            affinity=_affinity,
            threshold=clustering_threshold,
            method=clustering_method,
            supervised_scoring=b3_f_score),
        verbose=verbose,
        n_jobs=n_jobs).fit(X, y)

    labels = clusterer.labels_

    # Save predicted clusters
    if output_clusters:
        clusters = {}

        for label in np.unique(labels):
            mask = (labels == label)
            clusters[str(label)] = [r[0]["signature_id"] for r in X[mask]]

        json.dump(clusters, open(output_clusters, "w"))

    # Statistics
    if verbose and input_clusters:
        print("Number of blocks =", len(clusterer.clusterers_))
        print("True number of clusters", len(np.unique(y_true)))
        print("Number of computed clusters", len(np.unique(labels)))

        b3_overall = b3_precision_recall_fscore(y_true, labels)
        print("B^3 F-score (overall) =", b3_overall[2])

        if train_signatures_file:
            b3_train = b3_precision_recall_fscore(
                y_true[train_ids],
                labels[train_ids]
            )
            b3_test = b3_precision_recall_fscore(
                y_true[test_ids],
                labels[test_ids]
            )
            print("B^3 F-score (train) =", b3_train[2])
            print("B^3 F-score (test) =", b3_test[2])
            if results_file:
                paired_overall = paired_precision_recall_fscore(y_true, labels)
                paired_train = paired_precision_recall_fscore(
                    y_true[train_ids],
                    labels[train_ids]
                )
                paired_test = paired_precision_recall_fscore(
                    y_true[test_ids],
                    labels[test_ids]
                )

                json.dump({
                    "description": ["precision", "recall", "f_score"],
                    "b3": {"overall": list(b3_overall),
                           "train": list(b3_train),
                           "test": list(b3_test)
                           },
                    "paired": {"overall": list(paired_overall),
                               "train": list(paired_train),
                               "test": list(paired_test)
                               }
                }, open(results_file, 'w'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance_model", required=True, type=str)
    parser.add_argument("--input_signatures", required=True, type=str)
    parser.add_argument("--input_records", required=True, type=str)
    parser.add_argument("--input_clusters", default=None, type=str)
    parser.add_argument("--output_clusters", required=True, type=str)
    parser.add_argument("--clustering_method", default="average", type=str)
    parser.add_argument("--clustering_threshold", default=None, type=float)
    parser.add_argument("--train_signatures", default=None, type=str)
    parser.add_argument("--results_file", default=None, type=str)
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("--n_jobs", default=1, type=int)
    args = parser.parse_args()

    clustering(args.input_signatures, args.input_records, args.distance_model,
               args.input_clusters, args.output_clusters,
               args.verbose, args.n_jobs, args.clustering_method,
               args.train_signatures, args.clustering_threshold,
               args.results_file)
