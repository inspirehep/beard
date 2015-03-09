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
.. codeauthor:: Evangelos Tzemis <evangelos.tzemis@gmail.com>

"""

from __future__ import division
import numpy as np
import re


def _find_all(s, pattern):
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
            letters_cache[letter] = (tuple(_find_all(s1, letter)),
                                     tuple(_find_all(s2, letter)))

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
                if j - H <= i <= j + H:
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


def levenshtein(a, b):
    """Calculate the levenshtein distance between strings a and b.

    Case sensitiveness is activated, meaning that uppercase letters
    are treated differently than their corresponding lowercase ones.

    Parameters
    ----------
    :param a: string
        String to be compared

    :param b: string
        String to be compared

    Returns
    -------
    :returns int:
        The calculated levenshtein distance.
    """
    len_a, len_b = len(a), len(b)

    if len_a < len_b:
        return levenshtein(b, a)
    if len_b == 0:
        return len_a

    # We use tuple() to force strings to be used as sequences.
    a = np.array(tuple(a))
    b = np.array(tuple(b))

    # Instead of calculating the whole matrix, we only keep the last 2 rows.
    previous_row = np.arange(len_b + 1)
    for character in a:
        # Insertion
        current_row = previous_row + 1
        # Substitution or matching
        current_row[1:] = np.minimum(
            current_row[1:],
            np.add(previous_row[:-1], b != character))
        # Deletion
        current_row[1:] = np.minimum(
            current_row[1:],
            current_row[:-1] + 1)
        previous_row = current_row

    return current_row[-1]
