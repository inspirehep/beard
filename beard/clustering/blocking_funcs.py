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
from beard.utils.names import dm_tokenize_name
from beard.utils.names import first_name_initial


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
    of the double metaphone algorithm. Such signature might correspond, for
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

        # Note that in case where the cluster is created from a tuple of tokens
        # with multiple surnames, this signature will be counted to
        # _single_surname_signatures. This way we can omit dividing by 0.
        self._single_surname_signatures = 1
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

        if len(surnames) == 1:
            self._single_surname_signatures += 1

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

    # The bonus used in given_names_score. If there is a full name match
    # between signatures, this signature is counted FULL_NAME_MATCH_BONUS
    # times.
    FULL_NAME_MATCH_BONUS = 10

    def given_names_score(self, new_given_names, last_surname):
        """Count matches among the initials.

        A match is defined as match of all the given names from the new
        signature with the some of the names from the old signature.

        Such a match needs to keep order. For example ("A", "S") will match
        ("A", "D", "S"), but won't match ("S", "A", "D").

        In case of full given names available the match considers:
        + Full name match if both compared element are full names. Such a match
        is counted FULL_NAME_MATCH_BONUS times to the result, as it indicates
        the same author with bigger probability.
        + Initials match if none of compared elements is a full name.
        + Initials match if one od the compared elements is a full name.

        Two characters in initials - ``A`` and ``H`` are handled in special
        way in initial comparison, as the double metaphone can output an empty
        string for them.

        Parameters
        ----------
        :param new_given_names: tuple
            Strings representing given names of the new author
        :param last_surname: tuple
            Tokens, usually one, representing last surname(s) of the author.

        Raises
        ------
        :raises: KeyError
            When the last surname is not included in the cluster

        Returns
        -------
        :returns: integer
            Score for matching given names in the cluster.
        """
        result = 0.0
        if last_surname in self._content:
            for given_names, occurences in \
                    six.iteritems(self._content[last_surname]):
                given_names_length = len(given_names)
                old_names_index = 0
                names_match = self.FULL_NAME_MATCH_BONUS
                for new_name in new_given_names:
                    while old_names_index < given_names_length:
                        full_names_match = \
                            self._names_match(given_names[old_names_index],
                                              new_name)
                        if not full_names_match:
                            # There was no match, check next name from the
                            # old signature
                            old_names_index += 1
                        else:
                            if names_match:
                                names_match = full_names_match
                            # Go to the next name from the new ones.
                            break
                    if old_names_index == given_names_length:
                        # No match.
                        names_match = 0
                        break
                if names_match:
                    result += occurences * names_match * len(new_given_names) \
                        / float(len(given_names))
            return result
        self._raise_keyerror(last_surname)

    def single_surname_signatures(self):
        """Return the number of signatures with only one surname.

        The only exception is that when the block was created by a
        signature which consisted of more than one surname, the result will
        be increased by one, so that the result can be used as a denominator.

        Returns
        -------
        :returns: integer
            Number of signatures with only one surname or 1 if there are no
            such.
        """
        return self._single_surname_signatures

    def _names_match(self, name1, name2):

        if name1 == "" or name2 == "":
            # Probably starting with "h" or "w"
            if name1 != "":
                name1, name2 = name2, name1
            if name1 == name2:
                return True
            # Please not that names starting with "w" will result in strings
            # starting from "A" as the results of the double metaphone
            # algorithm.
            return name2.startswith('H') or name2.startswith('A')

        if len(name1) > 1 and len(name2) > 1:
            # Full names
            return self.FULL_NAME_MATCH_BONUS * (name1 == name2)

        # Just check initials
        return name1[0] == name2[0]

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
                                   first_name_initial(X[index
                                                        ][0]['author_name']))
        else:
            splitted_blocks.append(block._name)

    return splitted_blocks


def block_double_metaphone(X, threshold=1000):
    """Block the signatures.

    This blocking algorithm takes into consideration the cases, where
    author has more than one surname. Such a signature can be assigned
    to a block for the first author surname or the last one.

    The names are preprocessed by ``dm_tokenize_name`` function. As a result,
    here the algorithm operates on ``Double Metaphone`` tokens which are
    previously normalized.

    The algorithm has two phases. In the first phase, all the signatures with
    one surname are clustered together. Every different surname token creates
    a new block. In the second phase, the signatures
    with multiple surnames are compared with the blocks for the first and
    last surname.

    If the first surnames of author were already used as the last given names
    on some of the signatures, the new signature will be assigned to the block
    of the last surname.

    Otherwise, the algorithm check how many signatures have the same given
    names or initials for both considered blocks. The numbers are normalized
    using sizes of the blocks and compared with each other. The new signature
    is assigned to the block with bigger score.

    To prevent creation of too big clusters, the ``threshold`` parameter can
    be set. The algorithm will split every block which size is bigger than
    ``threshold`` into smaller ones using given names initials as the
    condition.

    The algorithm is order dependant, i.e. the order of signatures in the input
    can change the result. It happens in the case where there are more than
    one signatures with exactly the same combination of multiple surnames.
    The first signature is assigned to a block which matches it in the best
    way. Then, the rest of them are assigned to the same block without any
    scores computed.

    Parameters
    ----------
    :param X: numpy array
        Array of one element arrays of dictionaries. Each dictionary
        represents a signature. The algorithm needs ``author_name`` field in
        the dictionaries in order to work.
    :param threshold: integer
        Size above which the blocks will be split into smaller ones.

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
        tokens = dm_tokenize_name(signature_array['author_name'])
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
            last_metaphone_score = 0

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

                # No match, compute heuristically the match over initials

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

                # Second case is when the first surname is dropped.
                # A good example might be a woman who took her husband's
                # surname as the first one. Check how many names in the block
                # are the same. The score will be compared with a similar
                # approach for the other block.
                last_metaphone_score = \
                    cluster.given_names_score(given_names, (surnames[-1],)) / \
                    float(cluster.single_surname_signatures())

            except KeyError:
                # No such block
                pass

            try:
                # First surname one more time

                cluster = id_to_block[surnames[0]]

                # Check the case when the last surname is dropped.
                first_metaphone_score = 3 * \
                    cluster.given_names_score(given_names, (surnames[0],)) / \
                    float(cluster.single_surname_signatures())

                # Decide where the new signature should be assigned.
                if last_metaphone_score > first_metaphone_score:
                    id_to_block[surnames[-1]].add_signature(*tokens)
                    blocks.append(id_to_block[surnames[-1]])
                else:
                    cluster.add_signature(*tokens)
                    blocks.append(cluster)

                continue

            except KeyError:
                # No such block
                pass

            # No block for the first surname and no perfect match for the
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
        Array of one element arrays of dictionaries.

    Returns
    -------
    :returns: numpy array
        Array with ids of the blocks. As there is only one block, every element
        equals zero.
    """
    return np.zeros(len(X), dtype=np.int)


def block_last_name_first_initial(X):
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
