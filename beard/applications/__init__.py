# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Applications."""

from .clustering import clustering
from .learning_model import learn_model
from .utils import load_signatures
from .utils import get_author_full_name
from .utils import get_author_other_names
from .utils import get_author_initials
from .utils import get_author_affiliation
from .utils import get_title
from .utils import get_journal
from .utils import get_abstract
from .utils import get_coauthors
from .utils import get_keywords
from .utils import get_collaborations
from .utils import get_references
from .utils import get_year
from .utils import group_by_signature

__all__ = ("clustering",
           "learn_model",
           "load_signatures",
           "get_author_full_name",
           "get_author_other_names",
           "get_author_initials",
           "get_author_affiliation",
           "get_title",
           "get_journal",
           "get_abstract",
           "get_coauthors",
           "get_keywords",
           "get_collaborations",
           "get_references",
           "get_year",
           "group_by_signature")
