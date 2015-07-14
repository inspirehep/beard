# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Test cluster assignment.

.. codeauthor:: Mateusz Susik <mateusz.susik@gmail.com>

"""

from beard.matching.clusters import cluster_assignment

import pytest


@pytest.fixture
def simple_cost():
    """Simplest, constant cost function."""
    return lambda x, y, z, a: 1


@pytest.fixture
def intersection_cost():
    """The bigger the intersection is, the lower the cost."""
    return lambda x, y, z, a: -len(set(x).intersection(y))


@pytest.fixture
def weird_cost():
    """The bigger the intersection is, the bigger the cost."""
    return lambda x, y, z, a: -1 if len(set(x).intersection(y)) == 1 else 0


@pytest.fixture
def sensible_cost():
    """Cost function that takes into account difference and intersection."""
    return lambda x, y, z, a: -len(set(x).intersection(y)) + \
        len(set(y).difference(x))


def test_empty_matching(simple_cost):
    """Test the matching for no entities."""
    assert cluster_assignment({}, {}, simple_cost) == ({}, [])


def test_matching_without_entities(simple_cost):
    """Test the matching if there are no entites in the cluster."""
    assert cluster_assignment({1: [], 2: [2, 3, 4]}, {1: [2, 3, 4]},
                              simple_cost) == ({1: [], 2: [2, 3, 4]}, [])


def test_intersection_matching(intersection_cost):
    """Test more advanced cost function."""
    assert cluster_assignment({1: [1], 2: [2, 3]}, {1: [1, 2, 3]},
                              intersection_cost) == ({1: [], 2: [1, 2, 3]}, [])


def test_reversed_intersection_matching(weird_cost):
    """Test another cost function."""
    assert cluster_assignment({1: [1], 2: [2, 3]}, {1: [1, 2, 3]},
                              weird_cost) == ({2: [], 1: [1, 2, 3]}, [])


def test_claimed_paper(intersection_cost):
    """Test moving a claimed paper after reassignment."""
    assert cluster_assignment({1: [1, 2], 2: [3]}, {1: [1], 2: [2, 3]},
                              intersection_cost,
                              claims={1: [2]}) == ({1: [1, 2], 2: [3]}, [])


def test_rejected_paper(intersection_cost):
    """Test moving a rejected paper after reassignment."""
    assert cluster_assignment({1: [1, 2], 2: [3]}, {1: [1], 2: [2, 3]},
                              intersection_cost,
                              rejected={2: [2]}) == ({1: [1, 2], 2: [3]}, [])


def test_disconnected_graph(simple_cost):
    """Test the matching on a disconnected graph."""
    assert cluster_assignment({1: [9], 2: [10, 11], 3: [12, 13], 4: [14]},
                              {5: [10], 6: [9, 11], 8: [12, 14], 9: [13]},
                              simple_cost) == ({1: [9, 11], 2: [10],
                                                3: [12, 14], 4: [13]}, [])


def test_additional_old_clusters(simple_cost):
    """Matching algorithm must have a possibility of creating new clusters."""
    assert cluster_assignment({1: [1, 2, 3]}, {1: [1], 2: [2], 3: [3]},
                              simple_cost) == ({1: [1]}, [[2], [3]])
