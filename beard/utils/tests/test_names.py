# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Tests of personal names helpers.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

from ..names import normalize_name


def test_normalize_name():
    """Test of normalize_name"""
    assert normalize_name("Doe, John") == "doe john"
    assert normalize_name("Doe, J.") == "doe j"
    assert normalize_name("Doe, J") == "doe j"
    assert normalize_name("Doe-Foe, Willem") == "doefoe willem"
    assert normalize_name("Doe-Foe Willem") == "doe foe willem"
    assert normalize_name("Dupont, René") == "dupont rene"
    assert normalize_name("Dupont., René") == "dupont rene"
    assert normalize_name("Dupont, Jean-René") == "dupont jean rene"
    assert normalize_name("Dupont, René, III") == "dupont rene"
    assert normalize_name("Dupont, René, Jr.") == "dupont rene"
    assert normalize_name("Dupont, J.R.") == "dupont j r"
    assert normalize_name("Dupont, J.-R.") == "dupont j r"
    assert normalize_name("Dupont, J.-R.") == "dupont j r"
    assert normalize_name("Dupont") == "dupont"
    assert normalize_name("Dupont J.R.") == "dupont j r"
