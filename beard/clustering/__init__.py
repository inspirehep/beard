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
from .blocking_funcs import block_phonetic
from .blocking_funcs import block_last_name_first_initial
from .blocking_funcs import block_single
from .wrappers import ScipyHierarchicalClustering

__all__ = ("BlockClustering",
           "block_phonetic",
           "block_last_name_first_initial",
           "block_single",
           "ScipyHierarchicalClustering")
