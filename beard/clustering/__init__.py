# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Clustering algorithms."""

from .blocking import BlockClustering
from .blocking_funcs import block_double_metaphone
from .blocking_funcs import block_single
from .wrappers import ScipyHierarchicalClustering

__all__ = ("BlockClustering",
           "block_double_metaphone",
           "block_single",
           "ScipyHierarchicalClustering")
