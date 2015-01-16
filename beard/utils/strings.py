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

from unidecode import unidecode


def asciify(string):
    """Transliterate a string to ASCII."""
    string = string.decode("utf8", "ignore")
    string = unidecode(string)
    string = string.replace(u"[?]", u"").encode("ascii")

    return string
