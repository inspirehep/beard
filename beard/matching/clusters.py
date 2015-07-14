# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Blocking for clustering estimators.

.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

import munkres
import six


def cluster_assignment(old_clusters, new_clusters, similarity_function,
                       claims={}, rejected={}):
    """Match results of the disambiguation with the present clusters.

    In the usual case, the results of disambiguation will have to be integrated
    into a productional database. Before the run of the algorithm, the clusters
    might have been associated to records in the system. Thus, the results
    need to be matched to the existing records.

    This problem can be represented as an assignment problem in a bipartite
    graph. To solve it, the Hungarian algorithm is used.

    In the assignment problem one needs to specify the costs between vertices.
    Here the cost is computed by ``similarity_function`` which takes a single
    old cluster and a single new cluster as arguments. For every new cluster
    there is also an artificial empty old cluster inserted as there might be
    no good matches.

    Important to note is that one can specify claimed and rejected entities
    for records. After the run of the Hungarian algorithm claimed entities
    will be moved to clusters to which they belonged before and the same
    will happen to rejected entities if they were assigned by the algorithm to
    the clusters where they were rejected.

    Parameters
    ----------
    :old_clusters: dict
        Old clusters. The keys are ids of clusters in the systems
        (record ids) and the values are lists of entities belonging to these
        records before.

    :new_clusters: dict
        Result of the disambiguation.

    :similarity_function: function
        Cost function. Its signature is

        .. code-block:: python

           def similarity(new_cluster, old_cluster, claims, rejections)

        Where:

        - ``new_cluster`` is the list of entities from the new cluster
        - ``old_cluster`` is the list of entities from the old cluster
        - ``claims`` is the list of claims which were claimed by this record
          in the past.
        - ``rejections`` is the list of entities that should not belong to
          this record.

        Note that claimed and rejected entities might be reassigned after
        solving the assignment problem. There is no need to take care of it
        inside the cost function, although it still might be useful.

        Please remember that every new cluster will be also compared with an
        artificial empty cluster. The cost function should be aware of that.

        The function should return a number. Here is a simple example:

        .. code-block:: python

           def similarity(new_cluster, old_cluster, claims, rejections):
               return -len(set(new_cluster).intersection(old_cluster))

    :claims: dict
        Claimed entities. The keys are ids of clusters in the system and the
        values are lists of entitites claimed by these records before.

    :rejected: dict
        Claimed entities. The keys are ids of clusters in the system and the
        values are lists of entitites rejected by corresponding records before.

    Returns
    -------
    :returns: tuple
        The first element is a dictionary with record ids as keys and lists
        of entities as values. It is a result of matching. Some entities
        might not be matched and they are returned as the list of lists (list
        of clusters) - the second element in the returned tuple.
    """
    def graph_traversal(current_this_side, current_other_side,
                        reversed_this_side, reversed_other_side,
                        this_side_clusters, other_side_clusters,
                        this_side_visited, other_side_visited,
                        current_cluster):
        # Graph traversal in a bipartite graph
        current_this_side.append(current_cluster)
        this_side_visited[current_cluster] = True
        for sig in this_side_clusters[current_cluster]:
            if not other_side_visited[reversed_other_side[sig]]:
                # Notice the sides switch in the function call
                graph_traversal(current_other_side, current_this_side,
                                reversed_other_side, reversed_this_side,
                                other_side_clusters, this_side_clusters,
                                other_side_visited, this_side_visited,
                                reversed_other_side[sig])

    assigned = {k: [] for k in six.iterkeys(old_clusters)}
    unassigned = []

    # Variables used in the algorithm.
    # Set of all claimed entities.
    claims_set = {z for claim_list in six.itervalues(claims)
                  for z in claim_list}
    # Map from entity id to new cluster id.
    reversed_new = {v: k for k, va in six.iteritems(new_clusters) for v in va}
    # Map from old id to new cluster id.
    reversed_old = {v: k for k, va in six.iteritems(old_clusters) for v in va}
    # Map from the old cluster id to information whether it has been already
    # visited
    old_visited = {k: False for k in six.iterkeys(old_clusters)}
    # Map from the new cluster id to information whether it has been already
    # visited
    new_visited = {k: False for k in six.iterkeys(new_clusters)}
    # The ids of new clusters currently processed
    current_new = []
    # The ids of old clusters currently processed
    current_old = []

    # For every unvisited node, start o graph traversal in order to create the
    # smallest bipartite connected graph.
    for k, n in six.iteritems(new_clusters):
        if not new_visited[k]:
            # The algorithm didn't run on this graph. Dig.
            current_new.append(k)
            new_visited[k] = True
            for sig in n:
                if not old_visited[reversed_old[sig]]:
                    graph_traversal(current_old, current_new,
                                    reversed_old, reversed_new,
                                    old_clusters, new_clusters,
                                    old_visited, new_visited,
                                    reversed_old[sig])

        # current_new and current_old contain the smallest connected
        # bipartite graph.
        if current_new:
            # We add artificial empty old clusters. Some of the new clusters
            # might not correspond to the old clusters
            cost_matrix = [[0 for x in current_old + current_new]
                           for x in current_new]
            for i, new in enumerate(current_new):
                for j, old in enumerate(current_old):
                    cost_matrix[i][j] = \
                        similarity_function(new_clusters[new],
                                            old_clusters[old],
                                            claims.get(old, []),
                                            rejected.get(old, []))
                for j in range(len(current_new)):
                    # Here the cost for the artificial clusters is computed.
                    cost_matrix[i][j + len(current_old)] = \
                        similarity_function(new_clusters[new], [], [], [])

            m = munkres.Munkres()
            # Run the Hungarian algorithm
            assignment = m.compute(cost_matrix)

            for new, old in assignment:
                to_new_cluster = []
                for sig in new_clusters[current_new[new]]:
                    if sig in claims_set:
                        # If claimed, the signature belongs to the same cluster
                        # as before
                        assigned[reversed_old[sig]].append(sig)
                    elif old < len(old_clusters):
                        if sig in rejected.get(current_old[old], []):
                            # If rejected by this author, the signature belongs
                            # to the same cluster as before
                            assigned[reversed_old[sig]].append(sig)
                        else:
                            # Assign to this cluster
                            assigned[current_old[old]].append(sig)
                    else:
                        # The signature wasn't assigned to an old cluster
                        to_new_cluster.append(sig)
                if to_new_cluster:
                    unassigned.append(to_new_cluster)

        current_new = []
        current_old = []

    return assigned, unassigned
