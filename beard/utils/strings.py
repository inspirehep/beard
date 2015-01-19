# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Helper functions for strings.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

import unicodedata


def asciify(string):
    """Transliterate a string to ASCII."""
    try:
        string = string.decode("utf8")
    except:
        pass

    string = unicodedata.normalize("NFKD", string)
    string = string.encode("ascii", "ignore")
    string = string.decode("utf8")

    return string
