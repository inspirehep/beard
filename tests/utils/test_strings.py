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

from beard.utils.strings import asciify


def test_asciify():
    """Test of asciify."""
    assert asciify("") == ""
    assert asciify("foo") == "foo"
    assert asciify("bèård") == "beard"
    assert asciify("schröder") == "schroder"
