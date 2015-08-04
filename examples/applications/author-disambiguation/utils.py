# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Helpers for author disambiguation.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

import json

from beard.utils import given_name
from beard.utils import name_initials
from beard.utils import normalize_name
from beard.utils import given_name_initial


def load_signatures(signatures_filename, records_filename):
    """Load signatures from JSON files.

    Parameters
    ----------
    :param signatures_filename: string
        Path to the signatures file. The file should be in json format.

    :param records_filename: string
        Path to the records file. The file should be in json formaat.

    Returns
    -------
    :returns: tuple
        Signatures and records. Both are lists of dictionaries.
    """
    signatures = json.load(open(signatures_filename, "r"))
    records = json.load(open(records_filename, "r"))

    if isinstance(signatures, list):
        signatures = {s["signature_id"]: s for s in signatures}

    if isinstance(records, list):
        records = {r["publication_id"]: r for r in records}

    for signature_id, signature in signatures.items():
        signature["publication"] = records[signature["publication_id"]]

    return signatures, records


def get_author_full_name(s):
    """Get author full name from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Normalized author name
    """
    v = s["author_name"]
    v = normalize_name(v) if v else ""
    return v


def get_first_given_name(s):
    """Get author first given name from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Author's first given name
    """
    v = given_name(s["author_name"], 0)
    return v


def get_second_given_name(s):
    """Get author second given name from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Author's second given name
    """
    v = given_name(s["author_name"], 1)
    return v


def get_second_initial(s):
    """Get author second given name's initial from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Second given name's initial. Empty string in case it's not available.
    """
    v = given_name_initial(s["author_name"], 1)
    try:
        return v
    except IndexError:
        return ""


def get_author_other_names(s):
    """Get author other names from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Normalized other author names
    """
    v = s["author_name"]
    v = v.split(",", 1)
    v = normalize_name(v[1]) if len(v) == 2 else ""
    return v


def get_author_initials(s):
    """Get author initials from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Initials, not separated
    """
    v = s["author_name"]
    v = v if v else ""
    v = "".join(name_initials(v))
    return v


def get_author_affiliation(s):
    """Get author affiliation from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Normalized affiliation name
    """
    v = s["author_affiliation"]
    v = normalize_name(v) if v else ""
    return v


def get_title(s):
    """Get publication's title from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Title of the publication
    """
    v = s["publication"]["title"]
    v = v if v else ""
    return v


def get_journal(s):
    """Get journal's name from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Journal's name
    """
    v = s["publication"]["journal"]
    v = v if v else ""
    return v


def get_abstract(s):
    """Get author full name from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Normalized author name
    """
    v = s["publication"]["abstract"]
    v = v if v else ""
    return v


def get_coauthors(s):
    """Get coauthors from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Coauthors ids separated by a space
    """
    v = s["publication"]["authors"]
    v = " ".join(v)
    return v


def get_coauthors_from_range(s, range=10):
    """Get coauthors from the signature.

    Only the signatures from the range-neighbourhood of the given signature
    will be selected. Signatures on the paper are ordered (although they don't
    have to be sorted!), and the distance between signatures is defined
    as absolute difference of the indices.

    The function was introduced due to the high memory usage of
    a simple version.

    Parameters
    ----------
    :param s: dict
        Signature
    :param range: integer
        The maximum distance for the signatures between the author and his
        coauthor.

    Returns
    -------
    :returns: string
        Coauthors ids separated by a space
    """
    v = s["publication"]["authors"]
    try:
        index = v.index(s["author_name"])
        v = " ".join(v[max(0, index-range):min(len(v), index+range)])
        return v
    except ValueError:
        v = " ".join(v)
        return v


def get_keywords(s):
    """Get keywords from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Keywords separated by a space
    """
    v = s["publication"]["keywords"]
    v = " ".join(v)
    return v


def get_topics(s):
    """Get topics from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Topics separated by a space
    """
    v = s["publication"]["topics"]
    v = " ".join(v)
    return v


def get_collaborations(s):
    """Get collaborations from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Collaboations separated by a space
    """
    v = s["publication"]["collaborations"]
    v = " ".join(v)
    return v


def get_references(s):
    """Get references from the signature.
    Parameters
    ----------
    :param s: dict
        Signature
    Returns
    -------
    :returns: string
        Ids of references separated by a space
    """
    v = s["publication"]["references"]
    v = " ".join(str(r) for r in v)
    v = v if v else ""
    return v


def get_year(s):
    """Get year from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: int
        Year of publication if present on the signature, -1 otherwise
    """
    v = s["publication"]["year"]
    v = int(v) if v else -1
    return v


def group_by_signature(r):
    """Grouping function for ``PairTransformer``.

    Parameters
    ----------
    :param r: iterable
        signature in a singleton.

    Returns
    -------
    :returns: string
        Signature id
    """
    return r[0]["signature_id"]
