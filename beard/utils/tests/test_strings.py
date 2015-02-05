# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Tests of string helpers.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

from ..strings import asciify
from ..strings import normalize_personal_name


def test_asciify():
    """Test of asciify"""
    assert asciify("") == ""
    assert asciify("foo") == "foo"
    assert asciify("bèård") == "beard"
    assert asciify("schröder") == "schroder"


def test_normalize_personal_name():
    """Test of normalize_personal_name"""
    assert normalize_personal_name("Doe, John") == "doe john"
    assert normalize_personal_name("Doe, J.") == "doe j"
    assert normalize_personal_name("Doe, J") == "doe j"
    assert normalize_personal_name("Doe-Foe, Willem") == "doefoe willem"
    assert normalize_personal_name("Doe-Foe Willem") == "doe foe willem"
    assert normalize_personal_name("Dupont, René") == "dupont rene"
    assert normalize_personal_name("Dupont., René") == "dupont rene"
    assert normalize_personal_name("Dupont, Jean-René") == "dupont jean rene"
    assert normalize_personal_name("Dupont, René, III") == "dupont rene"
    assert normalize_personal_name("Dupont, René, Jr.") == "dupont rene"
    assert normalize_personal_name("Dupont, J.R.") == "dupont j r"
    assert normalize_personal_name("Dupont, J.-R.") == "dupont j r"
    assert normalize_personal_name("Dupont, J.-R.") == "dupont j r"
    assert normalize_personal_name("Dupont") == "dupont"
    assert normalize_personal_name("Dupont J.R.") == "dupont j r"
