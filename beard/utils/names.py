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

from ..ext.metaphone import dm

from .misc import memoize
from .strings import asciify

RE_NORMALIZE_LAST_NAME = re.compile("\s+|\-")
RE_NORMALIZE_OTHER_NAMES = re.compile("(,\s(i{1,3}|iv|v|vi|jr))|[\.'\-,\s]+")


@memoize
def normalize_name(name):
    """Normalize a personal name.

    A personal name is assumed to be formatted as "Last Name, Other Names".
    """
    name = asciify(name).lower()
    names = name.split(",", 1)

    if len(names) == 2:
        name = "%s, %s" % (RE_NORMALIZE_LAST_NAME.sub("", names[0]), names[1])
    else:
        name = names[0]  # Assume various other names only

    name = RE_NORMALIZE_OTHER_NAMES.sub(" ", name)
    name = name.strip()

    return name


@memoize
def name_initials(name):
    """Compute the set of initials of a given name."""
    return set([w[0] for w in name.split()])

RE_APOSTROPHES = re.compile('\'+')
RE_REMOVE_NON_CHARACTERS = re.compile('[^a-zA-Z\',\s]+')
DROPPED_AFFIXES = {'a', 'ab', 'am', 'ap', 'abu', 'al', 'auf', 'aus', 'bar',
                   'bath', 'bat', 'ben', 'bet', 'bin', 'bint', 'd', 'da',
                   'dall', 'dalla', 'das', 'de', 'degli', 'del', 'dell',
                   'della', 'dem', 'den', 'der', 'di', 'do', 'dos', 'du', 'e',
                   'el', 'i', 'ibn', 'im', 'jr', 'l', 'la', 'las', 'le',
                   'los', 'm', 'mac', 'mc', 'mhic', 'mic', 'o', 'ter', 'und',
                   'v', 'van', 'vom', 'von', 'zu', 'zum', 'zur'}


@memoize
def dm_tokenize_name(name):
    """Create Double Metaphone tokens from the string.

     Parameters
    ----------
    :param name: string
        Name of the author. Usually it should be in the format:
        surnames, first names.

    Returns
    -------
    :return: tuple
        The first element is a tuple with the tokens for surnames, the second
        is a tuple with the tokens for first names. The tuple always contains
        exactly two elements. Only the first results of the double metaphone
        algorithm are included in tuples.
    """
    tokens = tokenize_name(name)

    # Use double metaphone
    tokens = tuple(map(lambda x: tuple(map(lambda y: dm(y)[0], x)),
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
