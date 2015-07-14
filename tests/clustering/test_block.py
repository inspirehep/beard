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


def test_contains(block):
    """Test contains method."""
    assert block.contains(("ABC",))
    assert not block.contains(("DEF",))
