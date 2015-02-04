# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Scoring metrics."""

from .clustering import paired_precision_recall_fscore
from .clustering import paired_precision_score
from .clustering import paired_recall_score
from .clustering import paired_f_score
from .text import jaro
from .text import jaro_winkler
from .text import levenshtein

__all__ = ("paired_precision_recall_fscore",
           "paired_precision_score",
           "paired_recall_score",
           "paired_f_score",
           "jaro",
           "jaro_winkler",
           "levenshtein")
