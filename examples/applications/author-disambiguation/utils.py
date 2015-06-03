# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Helpers for ML applications."""

import json


from beard.utils import name_initials
from beard.utils import normalize_name


def load_signatures(signatures_filename, records_filename):
    """Load signatures from json fiels.

    Parameters
    ----------
    :param signatures_filename: string
        Path to the signatures file

    :param records_filename: string
        Path to the records file.

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
    :param s: string
        Signature

    Returns
    -------
    :returns: string
        Normalized author name
    """
    v = s["author_name"]
    v = normalize_name(v) if v else ""
    return v


def get_author_other_names(s):
    """Get author other names from the signature.

    Parameters
    ----------
    :param s: string
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
    :param s: string
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
    :param s: string
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
    :param s: string
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
    :param s: string
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
    :param s: string
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
    :param s: string
        Signature

    Returns
    -------
    :returns: string
        Coauthors ids separated by a space
    """
    v = s["publication"]["authors"]
    v = " ".join(v)
    return v


def get_keywords(s):
    """Get keywords from the signature.

    Parameters
    ----------
    :param s: string
        Signature

    Returns
    -------
    :returns: string
        Keywords separated by a space
    """
    v = s["publication"]["keywords"]
    v = " ".join(v)
    return v


def get_collaborations(s):
    """Get collaborations from the signature.

    Parameters
    ----------
    :param s: string
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
    :param s: string
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
    :param s: string
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
    :param r: signature in a singleton.

    Returns
    -------
    :returns: string
        Signature id
    """
    return r[0]["signature_id"]
