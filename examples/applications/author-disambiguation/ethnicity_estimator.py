import pandas as pd
import numpy as np
import gc
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

from beard.utils import normalize_name


def learn_lsvc_param(X, y):
    """Looking for best parameter for LinearSVC"""
    transformer = name_transformer()

    X_ = transformer.fit_transform(X)

    grid = GridSearchCV(LinearSVC(),
                        param_grid={"C": np.linspace(4.0, 5.0, 2)},
                        cv=StratifiedShuffleSplit(y, n_iter=3, test_size=0.25),
                        verbose=3).fit(X_, y)
    C = grid.best_params_['C']

    return C


def name_transformer():
    """Name transformer."""
    transformer = TfidfVectorizer(analyzer="char_wb",
                                  ngram_range=(1, 5),
                                  min_df=0.00005,
                                  dtype=np.float32,
                                  decode_error="replace")
    return transformer


def build_race_estimator(X, y, C):
    """build a model for estimating the ethnicity."""
    transformer = name_transformer()

    classifier = LinearSVC(C=C)

    estimator = Pipeline([("transformer", transformer),
                          ("classifier", classifier)]).fit(X, y)

    retrurn estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_datafile", required=True,  type=str)
    parser.add_argument("--output_race_estimator", default="race_estimator.pickle",  type=str)
    args = parser.parse_args()

    data = pd.read_csv(args.input_datafile)  # csv file {RACE| NAMELAST| NAMEFRST}
    y = data.RACE.values
    X = ["%s, %s" % (last, first) for last, first in zip(data.NAMELAST.values, data.NAMEFRST.values)]
    X = map(lambda x: normalize_name(x), X)

    gc.collect()

    C = learn_lsvc_param(X, y)

    race_estimator = build_race_estimator(X, y, C)

    pickle.dump(race_estimator,
                open(args.output_race_estimator, "w"),
                protocol=pickle.HIGHEST_PROTOCOL)
