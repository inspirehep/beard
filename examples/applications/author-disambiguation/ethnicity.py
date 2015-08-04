# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Author disambiguation -- Build an estimator for guessing an author ethnic
group from his name.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Hussein Al-Natsheh <hussein.al.natsheh@cern.ch>

"""

import argparse
import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from beard.utils import normalize_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_datafile", required=True, type=str)
    parser.add_argument("--output_ethnicity_estimator",
                        default="ethnicity_estimator.pickle", type=str)
    parser.add_argument("--C", default=4.0, type=float)
    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.input_datafile)
    y = data.RACE.values
    X = ["%s, %s" % (last, first) for last, first in zip(data.NAMELAST.values,
                                                         data.NAMEFRST.values)]
    X = [normalize_name(name) for name in X]

    # Train an estimator
    estimator = Pipeline([
        ("transformer", TfidfVectorizer(analyzer="char_wb",
                                        ngram_range=(1, 5),
                                        min_df=0.00005,
                                        dtype=np.float32,
                                        decode_error="replace")),
        ("classifier", LinearSVC(C=args.C))])
    estimator.fit(X, y)

    pickle.dump(estimator,
                open(args.output_ethnicity_estimator, "w"),
                protocol=pickle.HIGHEST_PROTOCOL)
