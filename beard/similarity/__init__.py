# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Similarity learning algorithms."""

from .pairs import PairTransformer
from .pairs import CosineSimilarity
from .pairs import AbsoluteDifference
from .pairs import JaccardSimilarity

__all__ = ("PairTransformer",
           "CosineSimilarity",
           "AbsoluteDifference",
           "JaccardSimilarity")
