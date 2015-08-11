# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""File which alows you to import whole algorithm."""

from distance import learn_model
from clustering import clustering
from sampling import pair_sampling

__all__ = ('learn_model', 'clustering', 'pair_sampling')
