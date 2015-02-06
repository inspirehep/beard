# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Helper functions for handling personal names.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

import re

from .misc import memoize
from .strings import asciify

RE_NORMALIZE_LAST_NAME = re.compile("\s+|\-")
RE_NORMALIZE_OTHER_NAMES = re.compile("(,\s(i{1,3}|iv|v|vi|jr))|[\.'\-,\s]+")


@memoize
def normalize_name(name):
    """Normalize a personal name.

    A personal name is assumed to be formatted as "Last Name, Other Names".
    """
    name = asciify(name).lower()
    names = name.split(",", 1)

    if len(names) == 2:
        name = "%s, %s" % (RE_NORMALIZE_LAST_NAME.sub("", names[0]), names[1])
    else:
        name = names[0]  # Assume various other names only

    name = RE_NORMALIZE_OTHER_NAMES.sub(" ", name)
    name = name.strip()

    return name


@memoize
def name_initials(name):
    """Compute the set of initials of a given name."""
    return set([w[0] for w in name.split()])
