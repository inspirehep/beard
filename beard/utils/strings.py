# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Helper functions for strings.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

import chardet
from unidecode import unidecode


def decode_to_unicode(string):
    """Decode a string to unicode, using the detected encoding."""
    return string.decode(chardet.detect(string)["encoding"], "ignore")


def asciify(string):
    """Transliterate a string to ASCII."""
    string = decode_to_unicode(string)
    string = unidecode(string)
    string = string.replace(u"[?]", u"").encode("ascii")

    return string
