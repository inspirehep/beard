# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

r"""Script for generating the training set.

It samples pairs of signatures labeled with 1 if they are of different authors
or 0 if they are of the same author.

Examples of command line use:

Sampling without blocking

python sampling.py --input_clusters big/clusters.json \
    --train_signatures train.json --output_pairs pairs.json --use_blocking 0

Sampling with blocking, without balancing

python sampling.py --input_clusters big/clusters.json \
    --train_signatures train.json --output_pairs pairs.json --input_balanced 0

Sampling with blocking, with balancing and smaller sample size.

python sampling.py --input_clusters big/clusters.json --sample_size 500000 \
    --train_signatures train.json --output_pairs pairs.json --input_balanced 1


.. codeauthor:: Hussein Al-Natsheh <hussein.al.natsheh@cern.ch>
.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>
"""

from __future__ import print_function

import argparse
import json
import math
import numpy as np
import random

from beard.clustering import block_phonetic
from beard.clustering import block_last_name_first_initial


def _noblocking_sampling(sample_size, train_signatures, clusters_reversed):
    pairs = []
    # Pairs dict will prevent duplicates
    pairs_dict = {}
    category_size = sample_size / 2
    negative = 0
    while negative < category_size:
        s1 = random.choice(train_signatures)['signature_id']
        s2 = random.choice(train_signatures)['signature_id']
        if s1 == s2:
            continue
        elif s1 > s2:
            s1, s2 = s2, s1
        s1_cluster = clusters_reversed[s1]
        s2_cluster = clusters_reversed[s2]
        if s1_cluster != s2_cluster:
            if negative < category_size:
                if s1 in pairs_dict:
                    if s2 in pairs_dict[s1]:
                        continue
                    pairs_dict[s1].append(s2)
                else:
                    pairs_dict[s1] = [s2]
                pairs.append((s1, s2, 1))
                negative += 1

    print("successfully sampled pairs from different authors")

    positive_pairs = []
    for i in range(100):
        print("sampling positive examples: %s out of 100 folds" % (i+1))
        some_signatures = random.sample(train_signatures,
                                        len(train_signatures)/20)
        for i, s1 in enumerate(some_signatures):
            for s2 in some_signatures[i+1:]:
                s1_id = s1['signature_id']
                s2_id = s2['signature_id']
                s1_cluster = clusters_reversed[s1_id]
                s2_cluster = clusters_reversed[s2_id]
                if s1_cluster == s2_cluster:
                    positive_pairs.append((s1_id, s2_id, 0))

        sampled = random.sample(positive_pairs, category_size/100)
        pairs += sampled
        for s1, s2, _ in sampled:
            if s1 > s2:
                s2, s1 = s1, s2
                if s1 in pairs_dict:
                    if s2 in pairs_dict[s1]:
                        continue
                    pairs_dict[s1].append(s2)
                else:
                    pairs_dict[s1] = [s2]

    print("successfully sampled pairs belonging to the same author")
    return pairs


def pair_sampling(blocking_function,
                  blocking_threshold,
                  blocking_phonetic_alg,
                  clusters_filename,
                  train_filename,
                  balanced=1, verbose=1,
                  sample_size=1000000,
                  use_blocking=1):
    """Sampling pairs from the ground-truth data.

    This function builds a pair dataset from claimed signatures.
    It gives the ability to specify the
    blocking function and whether the sampling would be balanced or not.

    Parameters
    ----------
    :param blocking_function: string
        must be a defined blocking function. Defined functions are:
        - "block_last_name_first_initial"
        - "block_phonetic"

    :param blocking_threshold: int or None
        It determines the maximum allowed size of blocking on the last name
        It can only be:
        -   None; if the blocking function is block_last_name_first_initial
        -   int; if the blocking function is block_phonetic
            please check the documentation of phonetic blocking in
            beard.clustering.blocking_funcs.py

    :param blocking_phonetic_alg: string or None
        If not None, determines which phonetic algorithm is used. Options:
        -  "double_metaphone"
        -  "nysiis" (only for Python 2)
        -  "soundex" (only for Python 2)

    :param clusters_filename: string
        Path to the input clusters (ground-truth) file

    :param train_filename: string
        Path to train set file

    :param balanced: boolean
        determines if the sampling would be balanced.
        The balance is defined as the same number of pairs with the same name
        on signature and pairs with different names. The balance is preserved
        both in the pairs belonging to one authors and in the pairs belonging
        to different authors. Note that if there are not enough pairs to
        satisfy the balance condition, some of the pairs will be replicated.

    :param verbose: boolean
        determines if some processing statistics would be shown

    :param sample_size: integer
        The desired sample size

    :param use_blocking: boolean
        determines if the signatures should be blocked before sampling

    Returns
    -------
    :returns: list
        list of signature pairs
    """
    # Load ground-truth
    true_clusters = json.load(open(clusters_filename, "r"))
    clusters_reversed = {v: k for k, va in true_clusters.iteritems()
                         for v in va}

    train_signatures = json.load(open(train_filename, "r"))

    if not use_blocking:
        return _noblocking_sampling(sample_size, train_signatures,
                                    clusters_reversed)

    train_signatures_ids = []
    for item in train_signatures:
        train_signatures_ids.append([item])

    train_signatures_ids = np.array(train_signatures_ids)

    if blocking_function == "block_last_name_first_initial":
        blocking = block_last_name_first_initial(train_signatures_ids)
    elif blocking_function == "block_phonetic" and blocking_threshold:
        blocking = block_phonetic(train_signatures_ids,
                                  blocking_threshold,
                                  blocking_phonetic_alg)
    else:
        raise ValueError("No such blocking strategy.")

    category_size = sample_size / 4

    blocking_dict = {}

    for index, b in enumerate(blocking):
        if b in blocking_dict:
            blocking_dict[b].append(index)
        else:
            blocking_dict[b] = [index]

    # 'd' stands for different, 's' stands for same, 'a' stands for author
    # 'n' stands for name
    dasn = []
    sasn = []
    sadn = []
    dadn = []

    for _, sig_s in blocking_dict.iteritems():

        for i, s1 in enumerate(sig_s):
            for s2 in sig_s[i+1:]:
                s1_id = train_signatures[s1]['signature_id']
                s2_id = train_signatures[s2]['signature_id']
                s1_name = train_signatures[s1]['author_name']
                s2_name = train_signatures[s2]['author_name']
                s1_cluster = clusters_reversed[s1_id]
                s2_cluster = clusters_reversed[s2_id]
                if s1_cluster == s2_cluster:
                    # Same author
                    if s1_name == s2_name:
                        sasn.append((s1_id, s2_id, 0))
                    else:
                        sadn.append((s1_id, s2_id, 0))
                else:
                    # Different authors
                    if s1_name == s2_name:
                        dasn.append((s1_id, s2_id, 1))
                    else:
                        dadn.append((s1_id, s2_id, 1))

    if balanced:

        if verbose:
            print("len of dasn:", len(dasn))
            print("len of sadn:", len(sadn))
            print("len of sasn:", len(sasn))
            print("len of dadn:", len(dadn))

        all_pairs = map(lambda x: int(math.ceil(
                        category_size/float(len(x)))) * x,
                        [dasn, sasn, sadn, dadn])
        pairs = reduce(lambda x, y: x + random.sample(y, category_size),
                       all_pairs, [])
    else:
        positive = sasn + sadn
        negative = dasn + dadn
        pairs = random.sample(positive,
                              sample_size/2) + random.sample(negative,
                                                             sample_size/2)

    return pairs

if __name__ == "__main__":
    # Parse command line arugments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_pairs", default="pairs.json", type=str)
    parser.add_argument("--input_clusters", default="clusters.json", type=str)
    parser.add_argument("--input_blocking_function",
                        default="block_last_name_first_initial", type=str)
    parser.add_argument("--input_blocking_threshold", default=None, type=int)
    parser.add_argument("--input_blocking_phonetic_alg", default=None,
                        type=str)
    parser.add_argument("--input_balanced", default=1, type=int)
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("--sample_size", default=1000000, type=int)
    parser.add_argument("--train_signatures", required=True, type=str)
    parser.add_argument("--use_blocking", default=1, type=int)

    args = parser.parse_args()

    pairs = pair_sampling(
        blocking_function=args.input_blocking_function,
        blocking_threshold=args.input_blocking_threshold,
        blocking_phonetic_alg=args.input_blocking_phonetic_alg,
        clusters_filename=args.input_clusters,
        train_filename=args.train_signatures,
        balanced=args.input_balanced,
        verbose=args.verbose,
        sample_size=args.sample_size,
        use_blocking=args.use_blocking
    )

    if args.verbose:
        print("number of pairs", len(pairs))

    json.dump(pairs, open(args.output_pairs, "w"))

    print("The sampled pairs file was successfully created")
