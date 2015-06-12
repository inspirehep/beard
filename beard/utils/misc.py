# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Miscellaneous helpers.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

from functools import wraps


def memoize(func):
    """Memoization function."""
    cache = {}

    @wraps(func)
    def wrap(*args, **kwargs):

        frozen = frozenset(kwargs.items())
        if (args, frozen) not in cache:
            cache[(args, frozen)] = func(*args, **kwargs)
        return cache[(args, frozen)]

    return wrap
