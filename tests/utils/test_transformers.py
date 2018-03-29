# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Tests of generic transformers.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from beard.utils.transformers import FuncTransformer
from beard.utils.transformers import Shaper


def test_func_transformer():
    """Test for FuncTransformer."""
    X = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int)

    def myfunc(v):
        return v + 1

    Xt = FuncTransformer(myfunc).fit_transform(X)
    assert_array_equal(Xt, X + 1)
    assert_equal(X.dtype, Xt.dtype)

    Xt = FuncTransformer(myfunc, dtype=np.float).fit_transform(X)
    assert_array_equal(Xt, X + 1)
    assert_equal(Xt.dtype, np.float)


def test_shaper():
    """Test for Shaper"""
    X = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int)

    Xt = Shaper((-1, 1)).fit_transform(X)
    assert_array_equal(Xt, [[0], [1], [2], [3], [4], [5]])
    assert_array_equal(Xt.shape, (6, 1))

    Xt = Shaper((-1,)).fit_transform(X)
    assert_array_equal(Xt, [0, 1, 2, 3, 4, 5])
    assert_array_equal(Xt.shape, (6,))

    Xt = Shaper((-1, 1), order="F").fit_transform(X)
    assert_array_equal(Xt, [[0], [3], [1], [4], [2], [5]])
    assert_array_equal(Xt.shape, (6, 1))
    # assert np.isfortran(Xt)
