# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Similarity learning algorithms."""

from .pairs import AbsoluteDifference
from .pairs import CosineSimilarity
from .pairs import EstimatorTransformer
from .pairs import ElementMultiplication
from .pairs import JaccardSimilarity
from .pairs import PairTransformer
from .pairs import StringDistance
from .pairs import Thresholder

__all__ = ("AbsoluteDifference",
           "CosineSimilarity",
           "EstimatorTransformer",
           "ElementMultiplication",
           "JaccardSimilarity",
           "PairTransformer",
           "StringDistance",
           "Thresholder")
