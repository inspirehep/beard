# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.


"""
This script is used for generating a training set for the distance model
It samples pairs of signatures labled with 1 if they are of different authors
or 0 if they are of the same author.

.. codeauthor:: Hussein Al-Natsheh <hussein.al.natsheh@cern.ch>
.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

from __future__ import print_function

import argparse
import json
import numpy as np
import random
import pickle

from sklearn.cross_validation import train_test_split

from beard.clustering import block_double_metaphone
from beard.clustering import block_last_name_first_initial


def unique_rows(rows_list):

    rows = np.array(rows_list)
    x = np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))
    b = np.ascontiguousarray(rows).view(x)
    _, idx = np.unique(b, return_index=True)

    return rows[idx]


def sampling_pairs(blocking_function,
                   blocking_threshold,
                   sample_size,
                   signatures_filename,
                   clusters_filename,
                   test_size, random_state,
                   balanced, verbose):
    """Sampling pairs from the ground-truth data. This function builds a pair
        dataset from claimed signatures. It gives the ability to specify the
        blocking function and whether the sampling would be balanced or not.

        Parameters
        ----------
        :param blocking_function: string
            must be a defined blocking function. Defined functions are:
            - block_last_name_first_initial
            - block_double_metaphone

        :param blocking_threshold: int or None
            It determines the maximum allowed size of blocking on the last name
            It can only be:
            -   None; if the blocking function is block_last_name_first_initial
            -   int; if the blocking function is block_double_metaphone
                please check the documentation of double_metaphone blocking in
                beard.clustering.blocking_funcs.py

        :param sample_size: integer
            The desired sample size

        :param signatures_filename: string
            The path where the input signatures file is

        :param clusters_filename: string
            The path where the input clusters (ground-truth) file is

        :param test_size: float
            The size of the test set: 0.0 < size < 1.0.
            It is used for train_test_split. Should be the same as in clustering

        :param random_state: integer
            a random_state value.
            It is used for train_test_split. Should be the same as in clustering

        :param balanced: boolean
            determines if the sampling would be balanced or uniformly random

        :param verbose: boolean
            determines if some processing statistics would be shown

        Returns
        -------
        :returns: list
            list of signature pairs
        """
    # Loading signatures
    sigs = json.load(open(signatures_filename, "r"))
    signatures = np.array([[x] for x in sigs])

    # Loading ground-truth
    true_clusters = json.load(open(clusters_filename, "r"))
    clusters_reversed = {v: k for k, va in true_clusters.iteritems() for v in va}

    train, test = train_test_split(
        np.arange(len(signatures)),
        test_size=test_size,
        random_state=random_state)

    train_d = []
    for item in signatures[train]:
        train_d.append(item[0]['signature_id'])

    train_d = np.array(train_d)

    if blocking_function == "block_last_name_first_initial":
        blocking = block_last_name_first_initial(signatures)
    elif blocking_function == "block_double_metaphone" and blocking_threshold:
        blocking = block_double_metaphone(signatures, blocking_threshold)
    else:
        raise ValueError("The blocking_function must be "
                         " either 'block_last_name_first_initial'"
                         " or 'block_double_metaphone' with passing blocking_threshold.")

    s = sample_size / 4
    sampling_param = int(round(np.sqrt(s) / 2))

    blocking_dict = {}

    for index, b in enumerate(blocking):
        if b in blocking_dict:
            blocking_dict[b].append(index)
        else:
            blocking_dict[b] = [index]

    dasn = []
    sasn = []
    sadn = []
    dadn = []

    for block, sig_s in blocking_dict.iteritems():
        sampled = np.intersect1d(sig_s, train_d, assume_unique=True)
        if len(sampled) > sampling_param:
            sampled = random.sample(sampled, sampling_param)

        for i, s1 in enumerate(sampled):
            for s2 in sampled[i+1:]:
                s1_name = sigs[s1]['author_name']
                s2_name = sigs[s2]['author_name']
                s1_cluster = clusters_reversed[s1]
                s2_cluster = clusters_reversed[s2]
                if s1_cluster == s2_cluster:
                    # Same author
                    if s1_name == s2_name:
                        sasn.append((s1, s2, 0))
                    else:
                        sadn.append((s1, s2, 0))
                else:
                    # Different authors
                    if s1_name == s2_name:
                        dasn.append((s1, s2, 1))
                    else:
                        dadn.append((s1, s2, 1))

    unique_dasn = unique_rows(dasn)
    if verbose:
        print ("len of unique(dasn):", len(unique_dasn))

    unique_sadn = unique_rows(sadn)
    if verbose:
        print ("len of unique(sadn):", len(unique_sadn))

    unique_sasn = unique_rows(sasn)
    if verbose:
        print ("len of unique(sasn):", len(unique_sasn))

    unique_dadn = unique_rows(dadn)
    if verbose:
        print ("len of unique(dadn):", len(unique_dadn))

    if balanced:
        if len(unique_dasn) < s:
            dasn_s = [unique_dasn[i] for i in np.random.choice(range(len(unique_dasn)), s)]
        else:
            dasn_s = [unique_dasn[i] for i in random.sample(range(len(unique_dasn)), s)]
        sadn_s = [unique_sadn[i] for i in random.sample(range(len(unique_sadn)), s)]
        sasn_s = [unique_sasn[i] for i in random.sample(range(len(unique_sasn)), s)]
        dadn_s = [unique_dadn[i] for i in random.sample(range(len(unique_dadn)), s)]

        pairs = dasn_s + sadn_s + sasn_s + dadn_s
    else:
        pairs_l = dasn + unique_sadn + unique_sasn + unique_dadn
        rand_sampl = random.sample(range(len(pairs_l)), sample_size)
        pairs = [pairs_l[i] for i in rand_sampl]

    return pairs

if __name__ == "__main__":
    # Parse command line arugments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_pairs", default="pairs.pickle", type=str)
    parser.add_argument("--input_signatures", default="signatures.json", type=str)
    parser.add_argument("--input_clusters", default="clusters.json", type=str)
    parser.add_argument("--input_blocking_function", default="block_last_name_first_initial", type=str)
    parser.add_argument("--input_blocking_threshold", default=None, type=int)
    parser.add_argument("--input_test_size", default=0.9, type=float)
    parser.add_argument("--input_sample_size", default=1000000, type=int)
    parser.add_argument("--input_random_state", default=42, type=int)
    parser.add_argument("--input_balanced", default=True, type=bool)
    parser.add_argument("--verbose", default=True, type=bool)

    args = parser.parse_args()

    pairs = sampling_pairs(blocking_function=args.input_blocking_function,
                           blocking_threshold=args.input_blocking_threshold,
                           sample_size=args.input_sample_size,
                           signatures_filename=args.input_signatures,
                           clusters_filename=args.input_clusters,
                           test_size=args.input_test_size,
                           random_state=args.input_random_state,
                           balanced=args.input_balanced,
                           verbose=args.verbose)

    unique_pairs = unique_rows(pairs)
    if args.verbose:
        print ("number of pairs", len(pairs))
        print ("number of unique pairs", len(unique_pairs))

    pickle.dump(pairs, open(args.output_pairs, "w"))

    print ("The sampled pairs file is successfully created")
