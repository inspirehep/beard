# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Tests of _Block class.

.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

import pytest

from beard.clustering.blocking_funcs import _Block


@pytest.fixture
def block():
    """Create a block for mr Abc, D. Vasquez."""
    return _Block(*(("ABC",), ("D", "VSQ")))


def test_add_signature(block):
    """Test adding signatures to the cluster."""
    assert block._content[("ABC",)][("D", "VSQ")] == 1
    block.add_signature(*(("ABC",), ("D", "VSQ")))
    assert block._content[("ABC",)][("D", "VSQ")] == 2
    block.add_signature(*(("ABC",), ("E",)))
    assert block._content[("ABC",)][("E",)] == 1
    block.add_signature(*(("ABD",), ("D", "VSQ",)))
    assert block._content[("ABD",)][("D", "VSQ")] == 1
    block.add_signature(*(("ABC", ""), ("D", "VSQ")))
    # Check handling of multiple surnames
    block.add_signature(*(("ABD", "EFG"), ("D", "VSQ",)))
    assert block._content[("ABD", "EFG")][("D", "VSQ")] == 1
    assert block._content[("ABC",)][("D", "VSQ")] == 2


def test_compare_tokens_from_last(block):
    """Test comparing tokens from the back."""
    assert block.compare_tokens_from_last(("VSQ",), ("ABC",))
    assert block.compare_tokens_from_last(("C", "D", "VSQ",), ("ABC",))
    with pytest.raises(KeyError) as excinfo:
        block.compare_tokens_from_last(("VSQ",), ("DEF"))
        assert "cluster doesn't contain a key" in str(excinfo.value)
    assert not block.compare_tokens_from_last(("VSD",), ("ABC",))
    assert not block.compare_tokens_from_last(("DGM", "VSQ"), ("ABC",))


def test_names_score(block):
    """Test initial scoring function."""
    block.add_signature(*(("ABC",), ("D", "VSQ")))
    assert block.given_names_score(("D",), ("ABC",)) == 1.0
    assert block.given_names_score(("D", "V"), ("ABC",)) == 2.
    assert block.given_names_score(("V",), ("ABC",)) == 1.0
    assert block.given_names_score(("VSQ",), ("ABC",)) == \
        1.0 * block.FULL_NAME_MATCH_BONUS
    assert block.given_names_score(("D", "VSQ"), ("ABC",)) == \
        2.0 * block.FULL_NAME_MATCH_BONUS
    assert block.given_names_score(("D", "V", "R"), ("ABC",)) == 0.0
    assert block.given_names_score(("D", "VR"), ("ABC",)) == 0.0
    block.add_signature(*(("ABC",), ("", "")))
    assert block.given_names_score(("",), ("ABC",)) == 0.5
    assert block.given_names_score(("H",), ("ABC",)) == 0.5

    # Check wrong key
    with pytest.raises(KeyError) as excinfo:
        block.given_names_score(("VSQ",), ("DEF"))
        assert "cluster doesn't contain a key" in str(excinfo.value)

    # Double metaphone algorithm can output empty string as the result
    block.add_signature(*(("ABC",), ("E", "")))
    assert block.given_names_score(("E", "A"), ("ABC",)) == 1.0

    # Test the correctness of the function for multiple entities in the
    # block
    block.add_signature(*(("ABC",), ("D")))
    assert block.given_names_score(("D", "V"), ("ABC",)) == 2.0
    assert block.given_names_score(("D",), ("ABC",)) == 2.0


def test_single_surname_signatures():
    """Test retrieving the number of signature."""
    block_ = _Block(*(("ABC",), ("D", "VSQ")))
    assert block_.single_surname_signatures() == 1
    block_.add_signature(*(("ABC",), ("D", "VSQ")))
    assert block_.single_surname_signatures() == 2
    block_.add_signature(*(("ABC", "DEF"), ("D", "VSQ")))
    assert block_.single_surname_signatures() == 2
    block_ = _Block(*(("ABC", "DEF"), ("D", "VSQ")))
    assert block_.single_surname_signatures() == 1
    block_.add_signature(*(("ABC", "DEF"), ("D", "VSQ")))
    assert block_.single_surname_signatures() == 1


def test_contains(block):
    """Test contains method."""
    assert block.contains(("ABC",))
    assert not block.contains(("DEF",))
