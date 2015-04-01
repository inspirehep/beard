# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Tests of personal names helpers.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

from beard.ext.metaphone import dm

from beard.utils.names import dm_tokenize_name
from beard.utils.names import first_name_initial
from beard.utils.names import name_initials
from beard.utils.names import normalize_name


def test_name_initals():
    """Test extracting name initials."""
    assert name_initials("Dupont, Jean-René") == set(['D', 'J'])


def test_normalize_name():
    """Test of normalize_name."""
    assert normalize_name("Doe, John") == "doe john"
    assert normalize_name("Doe, J.") == "doe j"
    assert normalize_name("Doe, J") == "doe j"
    assert normalize_name("Doe-Foe, Willem") == "doefoe willem"
    assert normalize_name("Doe-Foe Willem") == "doe foe willem"
    assert normalize_name("Dupont, René") == "dupont rene"
    assert normalize_name("Dupont., René") == "dupont rene"
    assert normalize_name("Dupont, Jean-René") == "dupont jean rene"
    assert normalize_name("Dupont, René, III") == "dupont rene"
    assert normalize_name("Dupont, René, Jr.") == "dupont rene"
    assert normalize_name("Dupont, J.R.") == "dupont j r"
    assert normalize_name("Dupont, J.-R.") == "dupont j r"
    assert normalize_name("Dupont") == "dupont"
    assert normalize_name("Dupont J.R.") == "dupont j r"


def test_dm_tokenize_name_simple():
    """Test of tokenize_name."""
    assert dm_tokenize_name("Doe, John") == ((dm(u"Doe")[0],),
                                             (dm(u"John")[0],))
    assert dm_tokenize_name("Doe, J.") == dm_tokenize_name(u"Doe, J")
    assert dm_tokenize_name("Doe-Foe, Willem") == ((dm(u"Doe")[0],
                                                    dm(u"Foe")[0]),
                                                   (dm(u"Willem")[0],))
    assert dm_tokenize_name("Dupont, René") == \
        dm_tokenize_name("Dupont., René")
    assert dm_tokenize_name("Dupont, Jean-René") == \
        ((dm(u"Dupont")[0],), (dm(u"Jean")[0], dm(u"Rene")[0]))
    assert dm_tokenize_name("Dupont, René, III") == \
        ((dm(u"Dupont")[0],), (dm(u"Rene")[0], dm(u"III")[0]))
    assert dm_tokenize_name("Dupont, René, Jr.") == \
        ((dm(u"Dupont")[0],), (dm(u"Rene")[0], dm(u"Jr")[0]))
    assert dm_tokenize_name("Dupont, J.R.") == \
        dm_tokenize_name("Dupont, J.-R.")
    assert dm_tokenize_name("Dupont") == ((dm(u"Dupont")[0],), ('',))
    assert dm_tokenize_name("Jean Dupont") == dm_tokenize_name("Dupont, Jean")


def test_dm_tokenize_name_with_soft_sign():
    """Test correct handling of the cyrillic soft sign."""
    assert dm_tokenize_name("Aref'ev, M.") == ((dm(u"Arefev")[0],),
                                               (dm(u"M")[0],))
    # If the following letter is uppercase, split
    assert dm_tokenize_name("An'Sun, J.") == ((dm(u"An")[0], dm(u"Sun")[0]),
                                              (dm(u"J")[0],))


def test_dm_tokenize_name_remove_common_affixes():
    """Test correct removal of the common affixes."""
    assert dm_tokenize_name("von und zu Hohenstein, F.") == \
        dm_tokenize_name("Hohenstein, F.")
    # If the name consists of only the common prefixes, don't drop it, as
    # it might actually be the correct surname.
    assert dm_tokenize_name("Ben, Robert") == ((dm(u"Ben")[0],),
                                               (dm(u"Robert")[0],))
    # Don't drop affixes among the first names.
    assert dm_tokenize_name("Robert, L. W.") == ((dm(u"Robert")[0],),
                                                 (dm(u"L")[0], dm(u"W")[0]))


def test_first_name_initial():
    """ Test the extraction of the first initial."""
    assert first_name_initial("Doe, John") == 'j'
    assert first_name_initial("Doe-Foe, Willem") == 'w'
    assert first_name_initial("Dupont, Jean-René") == 'j'
    assert first_name_initial("Dupont, René, III") == 'r'
    assert first_name_initial("Mieszko") == ''
    assert first_name_initial("John Doe") == 'j'
    assert first_name_initial("Dupont, .J") == 'j'
