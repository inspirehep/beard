# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Generic transformers for data manipulation.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class FuncTransformer(BaseEstimator, TransformerMixin):
    """Apply a given function element-wise."""

    def __init__(self, func, dtype=None):
        """Initialize.

        Parameters
        ----------
        :param func: callable
            The function to apply on each element.

        :param dtype: numpy dtype
            The type of the values returned by `func`.
            If None, then use X.dtype as dtype.
        """
        self.func = func
        self.dtype = dtype

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Apply `func` on all elements of X.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns Xt: array-like, shape (n_samples, n_features)
            The transformed data.
        """
        dtype = self.dtype
        if dtype is None:
            dtype = X.dtype

        vfunc = np.vectorize(self.func, otypes=[dtype])
        return vfunc(X)


class Shaper(BaseEstimator, TransformerMixin):
    """Reshape arrays."""

    def __init__(self, newshape, order="C"):
        """Initialize.

        Parameters
        ----------
        :param newshape: int or tuple
            The new shape of the array.
            See numpy.reshape for further details.

        :param order: {'C', 'F', 'A'}
            The index order.
            See numpy.reshape for further details.
        """
        self.newshape = newshape
        self.order = order

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Reshape X.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        :returns Xt: array-like, shape (self.newshape)
            The transformed data.
        """
        return X.reshape(self.newshape, order=self.order)
