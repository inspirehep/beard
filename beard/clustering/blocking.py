# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Blocking for clustering estimators.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

from __future__ import print_function

import numpy as np
import time

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import ClusterMixin
from sklearn.utils import column_or_1d

from .blocking_funcs import block_single


class _SingleClustering(BaseEstimator, ClusterMixin):
    def fit(self, X, y=None):
        self.labels_ = block_single(X)
        return self

    def partial_fit(self, X, y=None):
        self.labels_ = block_single(X)
        return self

    def predict(self, X):
        return block_single(X)


def _parallel_fit(fit_, partial_fit_, estimator, verbose, data_queue,
                  result_queue):
    """Run clusterer's fit function."""
    # Status can be one of: 'middle', 'end'
    # 'middle' means that there is a block to compute and the process should
    # continue
    # 'end' means that the process should finish as all the data was sent
    # by the main process
    status, block, existing_clusterer = data_queue.get()

    while status != 'end':

        b, X, y = block

        if len(X) == 1:
            clusterer = _SingleClustering()
        elif existing_clusterer and partial_fit_ and not fit_:
            clusterer = existing_clusterer
        else:
            clusterer = clone(estimator)

        if verbose > 1:
            print("Clustering %d samples on block '%s'..." % (len(X), b))

        if fit_ or not hasattr(clusterer, "partial_fit"):
            try:
                clusterer.fit(X, y=y)
            except TypeError:
                clusterer.fit(X)
        elif partial_fit_:
            try:
                clusterer.partial_fit(X, y=y)
            except TypeError:
                clusterer.partial_fit(X)

        result_queue.put((b, clusterer))
        status, block, existing_clusterer = data_queue.get()

    data_queue.put(('end', None, None))
    return


def _single_fit(fit_, partial_fit_, estimator, verbose, data):
    """Run clusterer's fit function."""
    block, existing_clusterer = data
    b, X, y = block

    if len(X) == 1:
        clusterer = _SingleClustering()
    elif existing_clusterer and partial_fit_ and not fit_:
        clusterer = existing_clusterer
    else:
        clusterer = clone(estimator)

    if verbose > 1:
        print("Clustering %d samples on block '%s'..." % (len(X), b))

    if fit_ or not hasattr(clusterer, "partial_fit"):
        try:
            clusterer.fit(X, y=y)
        except TypeError:
            clusterer.fit(X)
    elif partial_fit_:
        try:
            clusterer.partial_fit(X, y=y)
        except TypeError:
            clusterer.partial_fit(X)

    return (b, clusterer)


class BlockClustering(BaseEstimator, ClusterMixin):
    """Implements blocking for clustering estimators.

    Meta-estimator for grouping samples into blocks, within each of which
    a clustering base estimator is fit. This allows to reduce the cost of
    pairwise distance computation from O(N^2) to O(sum_b N_b^2), where
    N_b <= N is the number of samples in block b.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.

    blocks_ : ndarray, shape (n_samples,)
        Array of keys mapping input data to blocks.
    """

    def __init__(self, affinity=None, blocking="single", base_estimator=None,
                 verbose=0, n_jobs=1):
        """Initialize.

        Parameters
        ----------
        :param affinity: string or None
            If affinity == 'precomputed', then assume that X is a distance
            matrix.

        :param blocking: string or callable, default "single"
            The blocking strategy, for mapping samples X to blocks.
            - "single": group all samples X[i] into the same block;
            - "precomputed": use `blocks[i]` argument (in `fit`, `partial_fit`
              or `predict`) as a key for mapping sample X[i] to a block;
            - callable: use blocking(X)[i] as a key for mapping sample X[i] to
              a block.

        :param base_estimator: estimator
            Clustering estimator to fit within each block.

        :param verbose: int, default=0
            Verbosity of the fitting procedure.

        :param n_jobs: int
            Number of processes to use.
        """
        self.affinity = affinity
        self.blocking = blocking
        self.base_estimator = base_estimator
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _validate(self, X, blocks):
        """Validate hyper-parameters and input data."""
        if self.blocking == "single":
            blocks = block_single(X)
        elif self.blocking == "precomputed":
            if blocks is not None and len(blocks) == len(X):
                blocks = column_or_1d(blocks).ravel()
            else:
                raise ValueError("Invalid value for blocks. When "
                                 "blocking='precomputed', blocks needs to be "
                                 "an array of size len(X).")
        elif callable(self.blocking):
            blocks = self.blocking(X)
        else:
            raise ValueError("Invalid value for blocking. Allowed values are "
                             "'single', 'precomputed' or callable.")

        return X, blocks

    def _blocks(self, X, y, blocks):
        """Chop the training data into smaller chunks.

        A chunk is demarcated by the corresponding block. Each chunk contains
        only the training examples relevant to given block and a clusterer
        which will be used to fit the data.

        Returns
        -------
        :returns: generator
            Quadruples in the form of ``(block, X, y, clusterer)`` where
            X and y are the training examples for given block and clusterer is
            an object with a ``fit`` method.
        """
        unique_blocks = np.unique(blocks)

        for b in unique_blocks:
            mask = (blocks == b)
            X_mask = X[mask, :]
            if y is not None:
                y_mask = y[mask]
            else:
                y_mask = None
            if self.affinity == "precomputed":
                X_mask = X_mask[:, mask]

            yield (b, X_mask, y_mask)

    def _fit(self, X, y, blocks):
        """Fit base clustering estimators on X."""
        self.blocks_ = blocks

        if self.n_jobs == 1:
            blocks_computed = 0
            blocks_all = len(np.unique(blocks))

            for block in self._blocks(X, y, blocks):
                if self.partial_fit_ and block[0] in self.clusterers_:
                    data = (block, self.clusterers_[block[0]])
                else:
                    data = (block, None)

                b, clusterer = _single_fit(self.fit_, self.partial_fit_,
                                           self.base_estimator, self.verbose,
                                           data)

                if clusterer:
                    self.clusterers_[b] = clusterer

                if blocks_computed < blocks_all:
                    print("%s blocks computed out of %s" % (blocks_computed,
                                                            blocks_all))
                blocks_computed += 1
        else:
            try:
                from multiprocessing import SimpleQueue
            except ImportError:
                from multiprocessing.queues import SimpleQueue

            # Here the blocks will be passed to subprocesses
            data_queue = SimpleQueue()
            # Here the results will be passed back
            result_queue = SimpleQueue()

            for x in range(self.n_jobs):
                import multiprocessing as mp
                processes = []

                processes.append(mp.Process(target=_parallel_fit, args=(
                                 self.fit_, self.partial_fit_,
                                 self.base_estimator, self.verbose,
                                 data_queue, result_queue)))
                processes[-1].start()

            # First n_jobs blocks are sent into the queue without waiting
            # for the results. This variable is a counter that takes care of
            # this.
            presend = 0
            blocks_computed = 0
            blocks_all = len(np.unique(blocks))

            for block in self._blocks(X, y, blocks):
                if presend >= self.n_jobs:
                    b, clusterer = result_queue.get()
                    blocks_computed += 1
                    if clusterer:
                        self.clusterers_[b] = clusterer
                else:
                    presend += 1
                if self.partial_fit_:
                    if block[0] in self.clusterers_:
                        data_queue.put(('middle', block, self.clusterers_[b]))
                        continue

                data_queue.put(('middle', block, None))

            # Get the last results and tell the subprocesses to finish
            for x in range(self.n_jobs):
                if blocks_computed < blocks_all:
                    print("%s blocks computed out of %s" % (blocks_computed,
                                                            blocks_all))
                    b, clusterer = result_queue.get()
                    blocks_computed += 1
                    if clusterer:
                        self.clusterers_[b] = clusterer

            data_queue.put(('end', None, None))

            time.sleep(1)

        return self

    def fit(self, X, y=None, blocks=None):
        """Fit individual base clustering estimators for each block.

        Parameters
        ----------
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
                  or (n_samples, n_samples)
            Input data, as an array of samples or as a distance matrix if
            affinity == 'precomputed'.

        :param y: array-like, shape (n_samples, )
            Input labels, in case of (semi-)supervised clustering.
            Labels equal to -1 stand for unknown labels.

        :param blocks: array-like, shape (n_samples, )
            Block labels, if `blocking == 'precomputed'`.

        Returns
        -------
        :returns: self
        """
        # Validate parameters
        X, blocks = self._validate(X, blocks)

        # Reset attributes
        self.clusterers_ = {}
        self.fit_, self.partial_fit_ = True, False

        return self._fit(X, y, blocks)

    def partial_fit(self, X, y=None, blocks=None):
        """Resume fitting of base clustering estimators, for each block.

        This calls `partial_fit` whenever supported by the base estimator.
        Otherwise, this calls `fit`, on given blocks only.

        Parameters
        ----------
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
                  or (n_samples, n_samples)
            Input data, as an array of samples or as a distance matrix if
            affinity == 'precomputed'.

        :param y: array-like, shape (n_samples, )
            Input labels, in case of (semi-)supervised clustering.
            Labels equal to -1 stand for unknown labels.

        :param blocks: array-like, shape (n_samples, )
            Block labels, if `blocking == 'precomputed'`.

        Returns
        -------
        :returns: self
        """
        # Validate parameters
        X, blocks = self._validate(X, blocks)

        # Set attributes if first call
        if not hasattr(self, "clusterers_"):
            self.clusterers_ = {}

        self.fit_, self.partial_fit_ = False, True

        return self._fit(X, y, blocks)

    def predict(self, X, blocks=None):
        """Predict data.

        Parameters
        ----------
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        :param blocks: array-like, shape (n_samples, )
            Block labels, if `blocking == 'precomputed'`.

        Returns
        -------
        :returns: array-like, shape (n_samples)
            The labels.
        """
        # Validate parameters
        X, blocks = self._validate(X, blocks)

        # Predict
        labels = -np.ones(len(X), dtype=np.int)
        offset = 0

        for b in np.unique(blocks):
            # Predict on the block, if known
            if b in self.clusterers_:
                mask = (blocks == b)
                clusterer = self.clusterers_[b]

                pred = np.array(clusterer.predict(X[mask]))
                pred[(pred != -1)] += offset
                labels[mask] = pred
                offset += np.max(clusterer.labels_) + 1

        return labels

    @property
    def labels_(self):
        """Compute the labels assigned to the input data.

        Note that labels are computed on-the-fly.
        """
        labels = -np.ones(len(self.blocks_), dtype=np.int)
        offset = 0

        for b in self.clusterers_:
            mask = (self.blocks_ == b)
            clusterer = self.clusterers_[b]

            pred = np.array(clusterer.labels_)
            pred[(pred != -1)] += offset
            labels[mask] = pred
            offset += np.max(clusterer.labels_) + 1

        return labels
