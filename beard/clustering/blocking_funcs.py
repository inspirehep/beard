# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""The algorithms for blocking.

.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

import numpy as np
import six

from beard.utils import normalize_name
from beard.utils.names import phonetic_tokenize_name
from beard.utils.names import given_name_initial


class _Block:
    """Representation of a block.

    Block stores information about different variation of names and the
    quantities of their appearances on the papers.

    Example of a block _content:

    .. code:: python

        {
            ('JNS',): {
                ('P',): 2, ('P', 'PL'): 3, ('P', 'JH'): 2
            },
            ('JNS', 'SM0'): {
                ('PAL', 'JH'): 5, ('JH',): 3, ('SMN',): 2
            },
            ('RCN', 'JNS'): {
                ('A',): 34
            }
        }

    From the example above, one can see that the block stores information
    about 5 signatures of 'JNS' 'SM0', 'PAL' 'JH'. Those strings are results
    of the phonetic algorithm. Such signature might correspond, for
    example, to Jones-Smith, Paul John.
    """

    def __init__(self, surnames, given_names):
        """Create a block. Add given names from the first signature.

        Parameters
        ----------
        :param surnames: tuple
            Strings representing surnames on a signature.
        :param given_names: tuple
            Strings representing given names on a signature.
        """
        self._content = {surnames: {given_names: 1}}

        self._name = surnames[-1]

    def add_signature(self, surnames, given_names):
        """Add a signature to the block.

        Parameters
        ----------
        :param surnames: tuple
            Strings representing surnames on a signature.
        :param given_names: tuple
            Strings representing given_names on a signature.
        """
        if surnames in self._content:
            if given_names in self._content[surnames]:
                self._content[surnames][given_names] += 1
            else:
                self._content[surnames][given_names] = 1
        else:
            self._content[surnames] = {given_names: 1}

    def compare_tokens_from_last(self, first_surnames, last_surname):
        """Check if a part of the surname matches with given names in block.

        For example, ``Sanchez-Gomez, Juan`` can appear on a signature as
        ``Gomez, Juan Sanchez``. This function checks if there is a match
        between surnames like ``Sanchez`` and the given names in the block.
        In this case, a signature like ``Gomez, J. Sanchez`` will create a
        match, while ``Gomez, Juan S.`` won't.

        Full names have to match. Only the signatures with single surname
        are used for matching.

        Parameters
        ----------
        :param first_surnames: tuple
            Tokens which represent  few first surnames. In form of a tuple of
            strings.
        :param last_surname: tuple
            Tokens, usually one, representing last surname(s) of the author.

        Raises
        ------
        :raises: KeyError
            When the last name is not included in the cluster

        Returns
        -------
        :returns: boolean
            Information whether cluster contains this author if some of the
            first last names are treated as the last given names.
        """
        if last_surname in self._content:
            for given_names in six.iterkeys(self._content[last_surname]):
                given_names_left = len(given_names)
                for reversed_index, name in \
                        enumerate(reversed(first_surnames)):
                    if given_names_left == 0:
                        return True
                    elif given_names[-(reversed_index + 1)] != name:
                        break
                    given_names_left -= 1
                    if reversed_index == len(first_surnames) - 1:
                        return True
            return False
        self._raise_keyerror(last_surname)

    def contains(self, surnames):
        """Check if there is at least one signature with given surnames.

        Parameters
        ----------
        :param surnames: tuple
            Strings representing surnames on a signature.

        Returns
        -------
        :returns: boolean
            True if there is at least one sinature with given surnames.
        """
        return surnames in self._content

    def _raise_keyerror(self, key):
        raise KeyError("The cluster doesn't contain a key %s" % key)


def _split_blocks(blocks, X, threshold):
    splitted_blocks = []
    id_to_size = {}

    for block in blocks:
        if block._name in id_to_size:
            id_to_size[block._name] += 1
        else:
            id_to_size[block._name] = 1

    for index, block in enumerate(blocks):
        if id_to_size[block._name] > threshold:

            splitted_blocks.append(block._name +
                                   given_name_initial(X[index
                                                        ][0]['author_name']))
        else:
            splitted_blocks.append(block._name)

    return splitted_blocks


def block_phonetic(X, threshold=1000, phonetic_algorithm="double_metaphone"):
    """Block the signatures.

    This blocking algorithm takes into consideration the cases, where
    author has more than one surname. Such a signature can be assigned
    to a block for the first author surname or the last one.

    The names are preprocessed by ``phonetic_tokenize_name`` function. As a
    result, here the algorithm operates on ``Double Metaphone`` tokens which
    are previously normalized.

    The algorithm has two phases. In the first phase, all the signatures with
    one surname are clustered together. Every different surname token creates
    a new block. In the second phase, the signatures
    with multiple surnames are compared with the blocks for the first and
    last surname.

    If the first surnames of author were already used as the last given names
    on some of the signatures, the new signature will be assigned to the block
    of the last surname.

    Otherwise, the signature will be assigned to the block of
    the first surname.

    To prevent creation of too big clusters, the ``threshold`` parameter can
    be set. The algorithm will split every block which size is bigger than
    ``threshold`` into smaller ones using given names initials as the
    condition.

    Parameters
    ----------
    :param X: numpy array
        Array of one element arrays of dictionaries. Each dictionary
        represents a signature. The algorithm needs ``author_name`` field in
        the dictionaries in order to work.
    :param threshold: integer
        Size above which the blocks will be split into smaller ones.
    :param phonetic algorithm: string
        Which phonetic algorithm will be used. Options:
        -  "double_metaphone"
        -  "nysiis" (only for Python 2)
        -  "soundex" (only for Python 2)

    Returns
    -------
    :returns: numpy array
        Array with ids of the blocks. The ids are strings. The order of the
        array is the same as in the ``X`` input parameter.
    """
    # Stores all clusters. It is the only way to access them.
    # Every cluster can be accessed by the token that was used to create it.
    # It is the last token from the surnames tokens passed to the constructor.
    id_to_block = {}

    # List of tuples. Used as the in-between state of the algorithm between
    # the first and the second states. The tuple contain the block name
    # if the signature has been already blocked or None otherwise, and the
    # tokens.
    ordered_tokens = []

    # First phase.
    # Create blocks for signatures with single surname

    for signature_array in X[:, 0]:
        tokens = phonetic_tokenize_name(signature_array['author_name'],
                                        phonetic_algorithm=phonetic_algorithm)
        surname_tokens = tokens[0]
        if len(surname_tokens) == 1:
            # Single surname case
            surname = surname_tokens[0]
            if surname not in id_to_block:
                id_to_block[surname] = _Block(*tokens)
            else:
                id_to_block[surname].add_signature(*tokens)
            ordered_tokens.append((surname, tokens))
        else:
            # Multiple surnames
            ordered_tokens.append((None, tokens))

    # Second phase.
    # Assign every signature with multiple surnames to the block of the
    # first surname or the block of the last surname.

    blocks = []

    for token_tuple in ordered_tokens:

        if token_tuple[0] is not None:

            # There is already a block
            blocks.append(id_to_block[token_tuple[0]])

        else:

            # Case of multiple surnames
            tokens = token_tuple[1]
            surnames, given_names = tokens

            # Check if this combination of surnames was already included
            try:
                # First surname

                cluster = id_to_block[surnames[0]]
                if cluster.contains(surnames):
                    cluster.add_signature(*tokens)
                    blocks.append(cluster)
                    continue
            except KeyError:
                # No such block
                pass

            try:
                # Last surname

                cluster = id_to_block[surnames[-1]]
                if cluster.contains(surnames):
                    cluster.add_signature(*tokens)
                    blocks.append(cluster)
                    continue

                # # No match, compute heuristically the match over initials

                # Firstly, check if some of the surnames were used as the
                # last given names on some of the signatures.
                index = len(surnames) - 1
                match_found = False

                while index > 0:
                    token_prefix = surnames[:index]
                    if cluster.compare_tokens_from_last(token_prefix,
                                                        (surnames[-1],)):
                        cluster.add_signature(*tokens)
                        match_found = True
                        break
                    index -= 1

                if match_found:
                    # There was a full name match, so it must be the same
                    # author.
                    blocks.append(cluster)
                    continue

            except KeyError:
                # No such block
                pass

            try:
                # No match with last surname. Match with the first one.
                cluster = id_to_block[surnames[0]]
                cluster.add_signature(*tokens)
                blocks.append(cluster)

                continue

            except KeyError:
                # No such block
                pass

            # No block for the first surname and no good match for the
            # last surname.
            if surnames[-1] not in id_to_block:
                # Create new block.
                id_to_block[surnames[-1]] = _Block(*tokens)
            blocks.append(id_to_block[surnames[-1]])

    return np.array(_split_blocks(blocks, X, threshold))


def block_single(X):
    """Block the signatures into only one block.

    Parameters
    ----------
    :param X: numpy array
        Array of singletons of dictionaries.

    Returns
    -------
    :returns: numpy array
        Array with ids of the blocks. As there is only one block, every element
        equals zero.
    """
    return np.zeros(len(X), dtype=np.int)


def block_last_name_first_initial(X):
    """Blocking function using last name and first initial as key.

    The names are normalized before assigning to a block.

    Parameters
    ----------
    :param X: numpy array
        Array of singletons of dictionaries.

    Returns
    -------
    :returns: numpy array
        Array with ids of the blocks. The order of the
        array is the same as in the ``X`` input parameter.
    """
    def last_name_first_initial(name):
        names = normalize_name(name).split(" ", 1)

        try:
            name = "%s %s" % (names[0], names[1].strip()[0])
        except IndexError:
            name = names[0]

        return name

    blocks = []

    for signature in X[:, 0]:
        blocks.append(last_name_first_initial(signature["author_name"]))

    return np.array(blocks)
