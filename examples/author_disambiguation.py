# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Author disambiguation simplified example.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

import argparse
import cPickle
import numpy as np

from beard.clustering import BlockClustering
from beard.clustering import ScipyHierarchicalClustering
from beard.metrics import paired_f_score


def load_data(file_records, file_signatures, file_similarity):
    records = cPickle.load(open(file_records, "r"))
    signatures = cPickle.load(open(file_signatures, "r"))
    similarity = cPickle.load(open(file_similarity, "r")).astype(np.int8)

    signatures = {s["signature_id"]: s for s in signatures}
    records = {r["publication_id"]: r for r in records}

    for v in records.values():
        v["references"] = set(v["references"])
        v["signatures"] = set()

    for i, s in signatures.items():
        records[s["publication_id"]]["signatures"].add(i)

    return records, signatures, similarity


def last_name_first_initial(name):
    try:
        last_name, first_initial = name.split(",", 1)
        last_name = last_name.strip()
        first_initial = first_initial.strip()
        first_initial = first_initial[0]
        return "%s %s" % (last_name, first_initial)

    except:
        return name


if __name__ == "__main__":
    # Parse command line arugments
    parser = argparse.ArgumentParser()
    parser.add_argument("file_records", type=str)
    parser.add_argument("file_signatures", type=str)
    parser.add_argument("file_similarity", type=str)
    parser.add_argument("--n_jobs", default=1, type=int)
    parser.add_argument("--random_state", default=42, type=int)
    parser.add_argument("--verbose", default=0, type=int)
    args = parser.parse_args()

    # Load data
    records, signatures, similarity = load_data(args.file_records,
                                                args.file_signatures,
                                                args.file_similarity)

    # Block clustering
    X_ids = [k for k in signatures.keys()]
    blocks = [last_name_first_initial(v["author_name"])
              for v in signatures.values()]

    # wip ...
