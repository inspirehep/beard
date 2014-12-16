#!/bin/bash
# This file is part of Beard.
# Copyright (C) 2014 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

# This script is freely inspired from the Scikit-Learn integration scripts.
# https://github.com/scikit-learn/scikit-learn/blob/master/continuous_integration/install.sh
# License: 3-clause BSD

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

sudo apt-get update -qq
sudo apt-get install -qq libatlas3gf-base libatlas-dev

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Use the miniconda installer for faster download / install of conda
# itself
wget http://repo.continuum.io/miniconda/Miniconda-3.7.3-Linux-x86_64.sh  \
    -O miniconda.sh

chmod +x miniconda.sh && ./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n testenv --yes python=$PYTHON_VERSION pip \
    numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION scikit-learn=$SKLEARN_VERSION
source activate testenv

pip install coveralls pep257 pytest pytest-pep8 pytest-cov pytest-cache sphinx

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
