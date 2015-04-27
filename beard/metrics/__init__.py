# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014, 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Scoring metrics."""

from .clustering import b3_precision_recall_fscore
from .clustering import b3_precision_score
from .clustering import b3_recall_score
from .clustering import b3_f_score
from .clustering import paired_precision_recall_fscore
from .clustering import paired_precision_score
from .clustering import paired_recall_score
from .clustering import paired_f_score
from .clustering import silhouette_score
from .text import jaro
from .text import jaro_winkler
from .text import levenshtein

__all__ = ("b3_precision_recall_fscore",
           "b3_precision_score",
           "b3_recall_score",
           "b3_f_score",
           "paired_precision_recall_fscore",
           "paired_precision_score",
           "paired_recall_score",
           "paired_f_score",
           "silhouette_score",
           "jaro",
           "jaro_winkler",
           "levenshtein")
