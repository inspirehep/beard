# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Helper functions for handling personal names.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

import functools
import re
import sys

from .misc import memoize
from .strings import asciify

RE_NORMALIZE_WHOLE_NAME = re.compile("[^a-zA-Z,\s]+")
RE_NORMALIZE_OTHER_NAMES = re.compile("(,\s(i{1,3}|iv|v|vi|jr))|[\.'\-,\s]+")
RE_APOSTROPHES = re.compile('\'+')
RE_REMOVE_NON_CHARACTERS = re.compile('[^a-zA-Z\',\s]+')
DROPPED_AFFIXES = {'a', 'ab', 'am', 'ap', 'abu', 'al', 'auf', 'aus', 'bar',
                   'bath', 'bat', 'ben', 'bet', 'bin', 'bint', 'd', 'da',
                   'dall', 'dalla', 'das', 'de', 'degli', 'del', 'dell',
                   'della', 'dem', 'den', 'der', 'di', 'do', 'dos', 'ds', 'du',
                   'e', 'el', 'i', 'ibn', 'im', 'jr', 'l', 'la', 'las', 'le',
                   'los', 'm', 'mac', 'mc', 'mhic', 'mic', 'o', 'ter', 'und',
                   'v', 'van', 'vom', 'von', 'zu', 'zum', 'zur'}


@memoize
def normalize_name(name, drop_common_affixes=True):
    """Normalize a personal name.

    Parameters
    ----------
    :param name: string
        Name, formatted as "Last Name, Other Names".

    :param drop_common_affixes: boolean
        If the affixes like ``della`` should be dropeed.

    Returns
    -------
    :return: string
        Normalized name, formatted as "lastnames first names" where last names
        are joined.
    """
    name = asciify(name).lower()
    name = RE_NORMALIZE_WHOLE_NAME.sub(' ', name)
    names = name.split(",", 1)
    if not names:
        return ""
    if len(names) == 1:
        # There was no comma in the name
        all_names = names[0].split(" ")
        if len(all_names) > 1:
            # The last string should be the surname
            names = [all_names[-1], " ".join(all_names[:-1])]
        else:
            names = [all_names[0], ""]

    if drop_common_affixes:
        last_names = names[0].split(" ")
        without_affixes = list(filter(lambda x: x not in DROPPED_AFFIXES,
                               last_names))
        if len(without_affixes) > 0:
            names[0] = "".join(without_affixes)
    else:
        names[0] = re.sub('\s', '', names[0])

    name = "%s, %s" % (names[0], names[1])
    name = RE_NORMALIZE_OTHER_NAMES.sub(" ", name)
    name = name.strip()

    return name


@memoize
def name_initials(name):
    """Compute the set of initials of a given name."""
    return set([w[0] for w in name.split()])


@memoize
def phonetic_tokenize_name(name, phonetic_algorithm="double_metaphone"):
    """Create Double Metaphone tokens from the string.

     Parameters
    ----------
    :param name: string
        Name of the author. Usually it should be in the format:
        surnames, first names.

    :param phonetic algorithm: string
        Which phonetic algorithm will be used. Options:
        -  "double_metaphone"
        -  "nysiis" (only for Python 2)
        -  "soundex" (only for Python 2)

    Returns
    -------
    :return: tuple
        The first element is a tuple with the tokens for surnames, the second
        is a tuple with the tokens for first names. The tuple always contains
        exactly two elements. Only the first results of the double metaphone
        algorithm are included in tuples.
    """
    if sys.version[0] == '2':
        import fuzzy
        dm = fuzzy.DMetaphone()
        soundex = fuzzy.Soundex(5)
        phonetic_algorithms = {
            "double_metaphone": lambda y: dm(y)[0] or '',
            "nysiis": lambda y: fuzzy.nysiis(y),
            "soundex": lambda y: soundex(y)
        }
    else:
        from ..ext.metaphone import dm
        phonetic_algorithms = {
            "double_metaphone": lambda y: dm(y)[0]
        }

    tokens = tokenize_name(name)
    # Use double metaphone
    tokens = tuple(map(lambda x: tuple(map(lambda y: phonetic_algorithms[
        phonetic_algorithm](y), x)),
        tokens))

    return tokens


@memoize
def tokenize_name(name, handle_soft_sign=True, drop_common_affixes=True):
    """Normalize the name and create tokens from it.

     Parameters
    ----------
    :param name: string
        Name of the author. Usually it should be in the format:
        surnames, first names.
    :param handle_soft_sign: boolean
        Should the case of cyrillic soft sign be handled.
    :param drop_common_affixes: boolean
        Should the common affixes like ``von`` be dropped.

    Returns
    -------
    :return: tuple
        The first element is a tuple with surnames, the second
        is a tuple first names. The tuple always contains
        exactly two elements.
    """
    name = asciify(name)

    # Get rid of non character. Leave apostrophes as they are handled in a
    # different way.
    name = RE_REMOVE_NON_CHARACTERS.sub(' ', name)

    if handle_soft_sign:
        # Handle the "miagkii znak" in russian names.
        matches = re.findall(r"^([^',]*)'([a-z].*)", name)
        if matches:
            name = matches[0][0] + matches[0][1]

    # Remove apostrophes
    name = RE_APOSTROPHES.sub(' ', name)

    # Extract surname and name
    tokens = name.split(',')
    # If there are no first names, the default value is an empty string.
    tokens = [tokens[0], functools.reduce(lambda x, y: x+y, tokens[1:], '')]

    # Remove whitespaces and split both surnames and first-names
    tokens = list(map(lambda x: ' '.join(x.split()).lower().split(' '),
                      tokens))

    # Special case where there is no first name, i.e. there was no comma in
    # the signature.
    if tokens[1] == [''] and len(tokens[0]) > 1:
        # Probably the first string is the first name
        tokens = [tokens[0][1:], [tokens[0][0]]]
    elif tokens[1] == ['']:
        tokens = [[tokens[0][0]], [u'']]

    if drop_common_affixes:
        # Remove common prefixes
        without_affixes = list(filter(lambda x: x not in DROPPED_AFFIXES,
                                      tokens[0]))
        if len(without_affixes) > 0:
            tokens[0] = without_affixes

    return tokens

RE_CHARACTERS = re.compile('\w')


@memoize
def given_name_initial(name, index=0):
    """Get the initial from the first given name if available.

    Parameters
    ----------
    :param name: string
        Name of the author. Usually it should be in the format:
        surnames, first names.
    :param index: integer
        Which given name's initial should be returned. 0 for first, 1 for
        second, etc.

    Returns
    -------
    :return: string
        The given name initial. Asciified one character, lowercase if
        available, empty string otherwise.
    """
    try:
        asciified = asciify(name.split(",")[1]).lower().strip()
        names = asciified.split(" ")
        return RE_CHARACTERS.findall(names[index])[0]
    except IndexError:
        if index > 0:
            return ""
        split_name = name.split(" ")
        if len(split_name) > 1:
            # For example "John Smith", without comma. The first string should
            # indicate the first given name.
            asciified = asciify(split_name[0]).lower().strip()
            try:
                return RE_CHARACTERS.findall(asciified)[0]
            except IndexError:
                pass
        return ""


@memoize
def given_name(full_name, index):
    """Get a specific given name from full name.

    Parameters
    ----------
    :param full_name: string
        Name of the author. Usually it should be in the format:
        surnames, first names.
    :param index: integer
        Which given name should be returned. 0 for the first, 1 for the second,
        etc.

    Returns
    -------
    :return: string
        Given name or empty string if it is not available.
    """
    try:
        given_names = full_name.split(',')[1].strip()
        try:
            return given_names.split(' ')[index]
        except IndexError:
            return ""
    except IndexError:
        names = full_name.split(' ')
        try:
            return names[index]
        except IndexError:
            return ""
