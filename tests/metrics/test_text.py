# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Test text metrics.

.. codeauthor:: Petros Ioannidis <petros.ioannidis91@gmail.com>
.. codeauthor:: Evangelos Tzemis <evangelos.tzemis@gmail.com>

"""
from __future__ import generators

from numpy.testing import assert_almost_equal
import pytest
from pytest import mark

from beard.metrics.text import _find_all
from beard.metrics.text import _jaro_matching
from beard.metrics.text import jaro
from beard.metrics.text import jaro_winkler
from beard.metrics.text import levenshtein


@mark.parametrize('s, letter, occur',
                  (('MARTHA', 'A', (1, 5)),
                   ('DWAYNE', 'D', (0, )),
                   ('A', 'A', (0, )),
                   ('AABAA', 'AA', (0, 3)),
                   ('ABCD', 'D', (3, ))))
def test_find_all_normal_string(s, letter, occur):
    """Test find_all behaviour for average cases."""
    assert tuple(_find_all(s, letter)) == occur


@mark.parametrize('s, letter',
                  (('MARTHA', 'Z'),
                   ('', 'A')))
def test_find_all_none_string(s, letter):
    """Test find_all behaviour for empty cases."""
    with pytest.raises(StopIteration):
        assert next(_find_all(s, letter))


@mark.parametrize('s, letter',
                  ((set(), 'A'),
                   (dict(), 'A'),
                   (int(), 'A'),
                   (float(), 'A'),
                   (list(), 'A')))
def test_find_all_abnormal_string(s, letter):
    """Test find_all behaviour called with wrong objects."""
    with pytest.raises(TypeError):
        next(_find_all(s, letter))


@mark.parametrize('s1, s2, match',
                  (('MARTHA', 'MARHTA', (6, 2)),
                   ('DWAYNE', 'DUANE', (4, 0)),
                   ('DUANE', 'DWAYNE', (4, 0)),
                   ('MARHTA', 'MARTHA', (6, 2))))
def test_jaro_matching(s1, s2, match):
    """Test jaro_matching behaviour."""
    assert _jaro_matching(s1, s2) == match


@mark.parametrize('s1, s2, match',
                  (('MARTHA', 'MARHTA', 0.944),
                   ('DWAYNE', 'DUANE', 0.822),
                   ('ABCDEFG', 'ABCDEFG', 1.0),
                   ('', 'ABCDEFG', 0.0),
                   ('ABCDEFG', 'HIGKLMN', 0.0),
                   ('apple', 'apple', 1.0)))
def test_jaro(s1, s2, match):
    """Test jaro_similarity_metric behaviour."""
    assert_almost_equal(jaro(s1, s2), match, 3)


@mark.parametrize('s1, s2, match',
                  (('MARTHA', 'MARHTA', 0.961),
                   ('DWAYNE', 'DUANE', 0.84),
                   ('ABCDEFG', 'ABCDEFG', 1.0),
                   ('', 'ABCDEFG', 0.0),
                   ('ABCDEFG', 'HIGKLMN', 0.0)))
def test_jaro_winkler(s1, s2, match):
    """Test jaro_similarity_metric behaviour."""
    assert_almost_equal(jaro_winkler(s1, s2), match, 3)


@mark.parametrize('string_a, string_b, distance',
                  (('back', 'book', 2),
                   ('weight', 'height', 1),
                   ('Adam', 'Adams', 1),
                   ('YES', 'yes', 3),
                   ('weight', 'muchweigh', 5),
                   ('grand father', '', len('grand father')),
                   ('', 'grand father', len('grand father')),
                   (' ', ' ', 0),
                   ('', '', 0)))
def test_levenshtein(string_a, string_b, distance):
    """Test levenshtein_metric behaviour."""
    assert levenshtein(string_a, string_b) == distance
