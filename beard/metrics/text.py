# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Text metrics.

.. codeauthor:: Petros Ioannidis <petros.ioannidis91@gmail.com>

"""

from __future__ import division
import re


def find_all(s, pattern):
    """Find all occurences of the given pattern.

    Parameters
    ----------
    :param s: string
        String to be searched

    :param letter: string
        Substring we are searching for

    Returns
    -------
    :returns: generator
        A generator that holds the indexes of the patterns
    """
    for match in re.finditer(pattern, s):
        yield match.start()


def _jaro_matching(s1, s2):
    """Return the number of matching letters and transpositions.

    Parameters
    ----------
    :param s1: string
        First string

    :param s2: string
        Second string

    Returns
    -------
    :returns: (int, int)
        The number of matching letters and transpositions
    """
    H = min(len(s1), len(s2)) // 2

    letters_cache = {}
    matches = 0
    transpositions = 0
    s1_matching_letters = []
    s2_matching_letters = []
    s1_matched_positions = []
    s2_matched_positions = []

    for letter in s1:
        if letter not in letters_cache:
            letters_cache[letter] = (tuple(find_all(s1, letter)),
                                     tuple(find_all(s2, letter)))

    for letter, (s1_positions, s2_positions) in letters_cache.items():
        for i in s1_positions:
            for j in s2_positions:
                if i - H <= j <= i + H:
                    if j not in s2_matched_positions:
                        matches += 1
                        s2_matched_positions.append(j)
                    s1_matching_letters.append((i, letter))
                    break

    for letter, (s1_positions, s2_positions) in letters_cache.items():
        for j in s2_positions:
            for i in s1_positions:
                if i - H <= j <= i + H:
                    if i not in s1_matched_positions:
                        s1_matched_positions.append(i)
                    s2_matching_letters.append((j, letter))
                    break

    s1_matching_letters.sort()
    s2_matching_letters.sort()
    transpositions = len(tuple(filter(lambda x: x[0][1] != x[1][1],
                               zip(s1_matching_letters,
                                   s2_matching_letters))))

    return matches, transpositions


def jaro(s1, s2):
    """Return the Jaro similarity of the strings s1 and s2.

    Parameters
    ----------
    :param s1: string
        First string

    :param s2: string
        Second string

    Returns
    -------
    :returns: float
        Similarity of s1 and s2

    Reference
    ---------
    Jaro, M. A., "Advances in record-linkage methodology as applied to
    matching the 1985 census of Tampa, Florida", Journal of the American
    Statistical Association, 84:414-420, 1989.
    """
    if len(s1) == 0 or len(s2) == 0:
        return 0

    n_matches, n_transpositions = _jaro_matching(s1, s2)

    if n_matches == 0:
        return 0

    return 1 / 3 * (n_matches / len(s1) +
                    n_matches / len(s2) +
                    (n_matches - n_transpositions / 2) / n_matches)


def jaro_winkler(s1, s2, p=0.1):
    """Return the Jaro-Winkler similarity of the strings s1 and s2.

    Parameters
    ----------
    :param s1: string
        First string

    :param s2: string
        Second string

    Returns
    -------
    :returns: float
        Similarity of s1 and s2

    Reference
    ---------
    Winkler, W. E., "The state of record linkage and current research
    problems", Statistical Research Division, US Census Bureau. 1999.
    """
    jaro_distance = jaro(s1, s2)

    common_prefix = 0
    for s1_letter, s2_letter in zip(s1, s2):
        if s1_letter == s2_letter and common_prefix < 4:
            common_prefix += 1
        else:
            break

    return jaro_distance + p * common_prefix * (1 - jaro_distance)
